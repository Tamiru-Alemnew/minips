"""Dense synchronous baseline: torch DistributedDataParallel (gloo all-reduce), same model, data and schedule."""
import argparse
import json
import math
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))

from minips import config, data, env, evaluate, models  # noqa: E402


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": cfg.get("weight_decay", 5e-4)}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], momentum=cfg.get("momentum", 0.9),
    )


def lr_factor(c: int, warmup: int, total: int) -> float:
    """Same schedule as the Go server: linear warm-up, then cosine to zero."""
    if c < warmup:
        return (c + 1) / warmup
    if total <= warmup:
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * min((c - warmup) / (total - warmup), 1)))


def byte_counting_hook(counter: list):
    """Counts the bytes each rank sends in a ring all-reduce: 2(N-1)/N x bucket size."""

    def hook(state, bucket):
        n = dist.get_world_size()
        buf = bucket.buffer()
        counter[0] += 2 * (n - 1) / n * buf.numel() * buf.element_size()
        fut = dist.all_reduce(buf.div_(n), async_op=True).get_future()
        return fut.then(lambda f: f.value()[0])

    return hook


def worker(rank: int, n: int, cfg: dict, seed: int, port: int, out_dir: str):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=n)
    torch.set_num_threads(cfg.get("threads_per_worker", 1))
    torch.manual_seed(seed)
    model = models.build(cfg["model"])
    torch.manual_seed(seed * 1000 + rank)
    ddp = DDP(model)
    sent = [0.0]
    ddp.register_comm_hook(None, byte_counting_hook(sent))

    shard = data.shard(data.load(cfg["data"], train=True), n, rank)
    steps = config.steps_per_epoch(cfg, len(shard))
    total = cfg["epochs"] * steps
    stream = data.batches(shard, cfg["batch_size"], seed * 1000 + rank)
    opt = make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda c: lr_factor(c, cfg.get("warmup_epochs", 0) * steps, total))
    loss_fn = nn.CrossEntropyLoss()
    straggler = cfg.get("straggler") or {}
    slowdown = straggler.get("slowdown", 0.0) if straggler.get("worker") == rank else 0.0
    snaps = []

    dist.barrier()
    t_start = time.time()
    for c in range(total):
        x, y = next(stream)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(ddp(x), y)
        t0 = time.time()
        loss.backward()
        opt.step()
        sched.step()
        if slowdown:
            time.sleep(slowdown * (time.time() - t0))
        if rank == 0 and (c + 1) % steps == 0:
            snaps.append({"epoch": (c + 1) // steps, "time": time.time() - t_start, "loss": loss.item(),
                          "params": {k: v.detach().half().clone() for k, v in model.named_parameters()}})
    t_end = time.time()
    summary = {"rank": rank, "t_start": t_start, "t_end": t_end, "bytes_sent": sent[0], "steps": total}
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(summary))
    if rank == 0:
        torch.save({"final": {k: v.detach().clone() for k, v in model.named_parameters()}, "snaps": snaps},
                   Path(out_dir, "rank0.pt"))
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = config.load(args.config)
    n = cfg["workers"]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(n, cfg, args.seed, port, tmp), nprocs=n, join=True)
        summaries = [json.loads(Path(tmp, f"rank{r}.json").read_text()) for r in range(n)]
        saved = torch.load(Path(tmp, "rank0.pt"))

    train, test = data.load(cfg["data"], train=True), data.load(cfg["data"], train=False)
    model = models.build(cfg["model"])
    models.load_flat(model, {k: v.numpy() for k, v in saved["final"].items()})
    acc, loss = evaluate.final(model, train, test, cfg.get("bn_batches", 50))

    size = cfg.get("curve_test_size", 2000)
    curve_test = data.Split(test.x[:size], test.y[:size])
    curve = []
    for s in saved["snaps"]:
        models.load_flat(model, {k: v.float().numpy() for k, v in s["params"].items()})
        evaluate.recalibrate_bn(model, train, cfg.get("bn_batches", 20))
        curve.append({"epoch": s["epoch"], "time": s["time"], "acc": evaluate.accuracy(model, curve_test)[0],
                      "train_loss": s["loss"]})

    sent = sum(s["bytes_sent"] for s in summaries)
    result = {
        "name": cfg["name"], "system": "allreduce", "seed": args.seed, "config": cfg,
        "git_commit": env.git_commit(), "hardware": env.hardware(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": models.param_count(model), "steps": summaries[0]["steps"],
        "final_acc": acc, "final_loss": loss,
        "wallclock_s": max(s["t_end"] for s in summaries) - min(s["t_start"] for s in summaries),
        "bytes_total": sent, "bytes_note": "ring all-reduce bytes sent, summed over ranks (computed from bucket sizes)",
        "workers": summaries, "curve": curve,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"{result['name']} seed={args.seed} acc={acc:.4f} wall={result['wallclock_s']:.1f}s "
          f"bytes={sent / 2**20:.1f}MiB")


if __name__ == "__main__":
    main()
