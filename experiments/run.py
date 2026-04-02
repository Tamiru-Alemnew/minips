"""Run one MiniPS experiment: init weights, start the Go server and N workers, collect results."""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))

from minips import config, data, env, evaluate, models  # noqa: E402
from minips.client import PSClient  # noqa: E402
from minips.init import write  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def curve(cfg: dict, run_dir: Path, train: data.Split, test: data.Split) -> list[dict]:
    points = []
    curve_test = evaluate.subset(test, cfg.get("curve_test_size", 2000))
    for snap in sorted(run_dir.glob("snap_*.pt")):
        s = torch.load(snap)
        model = models.build(cfg["model"])
        models.load_flat(model, {k: v.float().numpy() for k, v in s["params"].items()})
        evaluate.recalibrate_bn(model, train, cfg.get("bn_batches", 20))
        acc, _ = evaluate.accuracy(model, curve_test)
        points.append({"epoch": s["epoch"], "time": s["time"], "acc": acc, "train_loss": s["loss"]})
    return points


def run(cfg_path: str, seed: int, out: Path, server_bin: Path) -> dict:
    cfg_path = str(Path(cfg_path).resolve())
    cfg = config.load(cfg_path)
    ps, n = cfg["ps"], cfg["workers"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        init = tmp / "init.bin"
        torch.manual_seed(seed)
        model = models.build(cfg["model"])
        write(model, str(init))

        train = data.load(cfg["data"], train=True)
        steps = config.steps_per_epoch(cfg, data.shard_len(cfg["data"], n))
        addr = f"127.0.0.1:{free_port()}"
        server = subprocess.Popen(
            [str(server_bin), "--addr", addr, "--init", str(init), "--workers", str(n),
             "--shards", str(ps.get("shards", 4)), "--mode", ps["mode"], "--staleness", str(ps.get("staleness", 0)),
             "--lr", str(cfg["lr"]), "--total-steps", str(cfg["epochs"] * steps),
             "--warmup-steps", str(cfg.get("warmup_epochs", 0) * steps),
             "--momentum", str(cfg.get("momentum", 0.9)), "--weight-decay", str(cfg.get("weight_decay", 5e-4))],
            stderr=open(tmp / "server.log", "w"),
        )
        try:
            workers = [
                subprocess.Popen(
                    [sys.executable, "-m", "minips.train", "--server", addr, "--worker-id", str(i),
                     "--config", cfg_path, "--seed", str(seed), "--out-dir", str(tmp)],
                    cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "worker")},
                )
                for i in range(n)
            ]
            codes = [w.wait() for w in workers]
            if any(codes):
                log = (tmp / "server.log").read_text()[-2000:]
                raise RuntimeError(f"worker exit codes {codes}; server log:\n{log}")
            client = PSClient(addr, 0)
            client.register()
            stats = client.stats()
            weights, _ = client.pull(cfg["epochs"] * steps)
            client.close()
        finally:
            server.terminate()
            server.wait()

        summaries = [json.loads((tmp / f"worker{i}.json").read_text()) for i in range(n)]
        test = data.load(cfg["data"], train=False)
        models.load_flat(model, weights)
        acc, loss = evaluate.final(model, train, test, cfg.get("bn_batches", 50))
        points = curve(cfg, tmp, train, test)

    w = stats["workers"]
    result = {
        "name": cfg["name"], "system": "ps", "seed": seed, "config": cfg,
        "git_commit": env.git_commit(), "hardware": env.hardware(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": models.param_count(model), "steps": cfg["epochs"] * steps,
        "final_acc": acc, "final_loss": loss,
        "wallclock_s": max(s["t_end"] for s in summaries) - min(s["t_start"] for s in summaries),
        "bytes_push": sum(x["bytes_in"] for x in w), "bytes_pull": sum(x["bytes_out"] for x in w),
        "bytes_total": sum(x["bytes_in"] + x["bytes_out"] for x in w),
        "mean_wait_s": sum(x["wait_seconds"] for x in w) / n,
        "staleness_hist": stats["staleness_hist"], "updates_applied": stats["updates_applied"],
        "server_stats": w, "workers": summaries, "curve": points,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-bin", default=str(ROOT / "bin" / "ps-server"))
    args = ap.parse_args()
    r = run(args.config, args.seed, Path(args.out), Path(args.server_bin))
    print(f"{r['name']} seed={r['seed']} acc={r['final_acc']:.4f} wall={r['wallclock_s']:.1f}s "
          f"bytes={r['bytes_total'] / 2**20:.1f}MiB")


if __name__ == "__main__":
    main()
