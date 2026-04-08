# MiniPS design notes

The README has the short version. This is the long one.

## Process layout

```
              ┌──────────────── Go parameter server ────────────────┐
              │  gRPC handlers ──► consistency policy (vector clock) │
 worker 0 ──► │        │                                              │
 worker 1 ──► │        ▼                                              │
   ...        │  shard 0 goroutine   shard 1 goroutine   ...          │
 worker N ──► │  (weights, momentum, accumulator for its tensors)     │
              │  stats handler: wire bytes per worker                 │
              └──────────────────────────────────────────────────────┘
```

Every worker runs the same loop for clock `c = 0, 1, 2, ...`:

1. `Pull(c)`: blocks until the consistency policy allows a read at clock `c`, then streams every tensor.
   Under SSP the worker skips this call while its cached copy is still fresh enough (see below).
2. Forward and backward pass on one local batch.
3. Encode each gradient (dense, top-k or int8) and `Push` it as one client stream.
4. `Clock(c)`: tells the server iteration `c` is finished.

## Wire format

Each `GradChunk` and `ParamChunk` carries a tensor name, an encoding tag and raw little-endian bytes. A
tensor larger than 4 MiB is split across several messages, so no single gRPC message gets near the
default size limits. Tensors are never sent as `repeated float`: protobuf would encode each element
separately and the server would spend its time decoding.

| encoding | layout | bytes for n elements, k kept |
|---|---|---|
| `DENSE_F32` | n × float32 | 4n |
| `TOPK` | uint32 k, k × uint32 index, k × float16 value | 4 + 6k |
| `INT8` | float32 scale, n × int8 | 4 + n |

## Server internals

**Sharding.** Tensors go to shards by FNV-1a hash of their name, so placement is stable across runs
and needs no coordination. Each shard is one goroutine that owns its tensors' weights, momentum
buffers and BSP accumulators. Handlers send it requests on a channel, so no memory is ever written by
two goroutines and the update path holds no locks. Decoding a payload happens in the handler goroutine
before the request is sent, so decoding runs in parallel across workers.

**Optimiser.** SGD with momentum 0.9 and weight decay 5e-4 on weight matrices only (biases and
BatchNorm parameters get none), applied on the shard. Learning rate is linear warm-up then cosine to
zero, computed from the clock attached to each push.

**BSP.** Pushes for clock `c` are added into per-tensor accumulators. When the last worker calls
`Clock(c)`, the server flushes every shard (one averaged step), marks clock `c` applied and wakes all
readers waiting for `c + 1`.

**SSP with bound s.** The server keeps `done[w]`, the number of iterations each worker has finished. A
read at clock `c` is allowed while `c − min(done) ≤ s`; otherwise the handler waits on a condition
variable that is broadcast whenever any worker finishes an iteration. Gradients are applied as they
arrive, each scaled by `1/N` so that one round of N updates moves the weights about as far as one BSP
step.

Workers exploit the bound on their side too: a pull returns `min(done)` at the time of the read, and a
worker reuses its cached weights while `c − cached_min ≤ s`. Those weights already contain every update
from iterations before `cached_min`, which is exactly what the SSP guarantee promises, so skipping the
pull is safe and is where most of SSP's saving in pull traffic comes from.

**ASP.** Reads never block and updates apply on arrival. The staleness histogram shows how far behind
the slowest worker each read was.

**Byte counting.** A gRPC stats handler adds `WireLength` of every inbound and outbound payload to the
counters of the worker named in the `worker-id` request header. These are the framed message bytes gRPC
actually sent, not tensor sizes; HTTP/2 frame headers and TLS (not used here) are not included.

## Worker internals

**Top-k with error feedback.** For each tensor, the residual from the previous step is added to the
gradient, the k largest magnitudes are sent, and the new residual is everything not sent, including the
float16 rounding error of the values that were sent. So `sent + residual == gradient + old residual`
exactly; nothing is lost, only delayed. Without error feedback the dropped 99% is gone for good (the
`bsp_topk_noef` run shows the cost).

**Int8.** Per-tensor scale `max|g| / 127`, round to nearest; the error is at most half a step.

**BatchNorm statistics** are buffers, not gradients, so they stay local to each worker. Evaluation
assembles a model from the server's weights and recomputes BatchNorm statistics on training batches
before measuring test accuracy. The all-reduce baseline is evaluated the same way.

## Experiments

`experiments/run.py` writes the initial weights, starts the server and N workers as separate processes,
waits, asks the server for its counters, pulls the final weights and evaluates them. Wall-clock is from
the first worker's first pull to the last worker's last `Clock`, so process start-up and data loading are
excluded. Worker 0 saves its weights at every epoch boundary; those snapshots become the accuracy curve.

`baselines/allreduce.py` trains the same model on the same shards with the same schedule using
`torch.distributed` DDP over gloo. Its byte count comes from a DDP communication hook that adds
`2(N−1)/N × bucket bytes` per rank per step, which is what a ring all-reduce sends. That number is
computed, not measured on the wire, and the README says so.

## Failure handling

None. If a worker dies under BSP, the server waits for its `Clock` forever; under SSP the others stop
once they are `s` clocks ahead. A real system would need heartbeats, a timeout that drops the worker
from the vector clock, and server checkpoints so a restarted worker can rejoin.
