import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from . import compress, config, data, models
from .client import PSClient


def check_specs(model: nn.Module, specs) -> None:
    ours = [(n, list(p.shape)) for n, p in model.named_parameters()]
    theirs = [(s.name, list(s.shape)) for s in specs]
    if ours != theirs:
        raise SystemExit(f"model does not match server params: {ours[:3]}... vs {theirs[:3]}...")


def main():
    ap = argparse.ArgumentParser(description="One MiniPS training worker.")
    ap.add_argument("--server", default="127.0.0.1:50051")
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = config.load(args.config)
    torch.set_num_threads(cfg.get("threads_per_worker", 1))
    torch.manual_seed(args.seed * 1000 + args.worker_id)
    out = Path(args.out_dir)

    client = PSClient(args.server, args.worker_id)
    client.wait_ready()
    reg = client.register()
    n, mode, staleness = reg.num_workers, reg.mode, reg.staleness

    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = models.build(cfg["model"]).to(device)
    check_specs(model, client.specs)
    shard = data.load(cfg["data"], train=True, num_shards=n, index=args.worker_id)
    steps = config.steps_per_epoch(cfg, data.shard_len(cfg["data"], n))
    total = cfg["epochs"] * steps
    stream = data.batches(shard, cfg["batch_size"], args.seed * 1000 + args.worker_id)
    compressor = compress.make(cfg.get("compression"))
    params = list(model.named_parameters())
    loss_fn = nn.CrossEntropyLoss()

    straggler = cfg.get("straggler") or {}
    slowdown = straggler.get("slowdown", 0.0) if straggler.get("worker") == args.worker_id else 0.0
    snapshot = args.worker_id == 0 and cfg.get("snapshot_every_epoch", True)

    cached_min = None
    t_pull = t_compute = t_push = 0.0
    pulls = 0
    model.train()
    t_start = time.time()
    for c in range(total):
        t0 = time.time()
        if mode != "ssp" or cached_min is None or c - cached_min > staleness:
            weights, cached_min = client.pull(c)
            models.load_flat(model, weights)
            pulls += 1
        t1 = time.time()
        x, y = (t.to(device, non_blocking=True) for t in next(stream))
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        grads = [(name, *compressor.encode(name, p.grad)) for name, p in params]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        client.push(c, grads)
        client.clock(c)
        t3 = time.time()
        t_pull, t_compute, t_push = t_pull + t1 - t0, t_compute + t2 - t1, t_push + t3 - t2
        if slowdown:
            time.sleep(slowdown * (t2 - t1))
        if snapshot and (c + 1) % steps == 0:
            state = {name: p.detach().half().cpu() for name, p in params}
            torch.save({"epoch": (c + 1) // steps, "time": time.time() - t_start, "loss": loss.item(), "params": state},
                       out / f"snap_{(c + 1) // steps:03d}.pt")

    summary = {
        "worker_id": args.worker_id, "t_start": t_start, "t_end": time.time(), "steps": total,
        "steps_per_epoch": steps, "pulls": pulls,
        "pull_s": t_pull, "compute_s": t_compute, "push_s": t_push, "final_loss": loss.item(),
    }
    (out / f"worker{args.worker_id}.json").write_text(json.dumps(summary))
    client.close()


if __name__ == "__main__":
    main()
