"""Runs every config in experiments/configs/<matrix>/ for each seed; one JSON per run in results/<matrix>/."""
import argparse
import fnmatch
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = {"ps": ROOT / "experiments" / "run.py", "allreduce": ROOT / "baselines" / "allreduce.py"}
DEFAULT_SEEDS = {"full": 3, "smoke": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="full", choices=sorted(DEFAULT_SEEDS))
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--only", default="*", help="glob over run names, e.g. 'bsp_*'")
    ap.add_argument("--force", action="store_true", help="rerun even if the result file exists")
    args = ap.parse_args()

    seeds = range(args.seeds or DEFAULT_SEEDS[args.matrix])
    out_dir = ROOT / "results" / args.matrix
    configs = sorted((ROOT / "experiments" / "configs" / args.matrix).glob("*.yaml"))
    todo = [(c, s) for c in configs if fnmatch.fnmatch(c.stem, args.only) for s in seeds]
    failed = []
    for i, (cfg, seed) in enumerate(todo, 1):
        out = out_dir / f"{cfg.stem}_s{seed}.json"
        if out.exists() and not args.force:
            print(f"[{i}/{len(todo)}] {out.name} exists, skipping")
            continue
        system = yaml.safe_load(cfg.read_text())["system"]
        print(f"[{i}/{len(todo)}] {cfg.stem} seed={seed}", flush=True)
        t = time.time()
        code = subprocess.run([sys.executable, str(SCRIPTS[system]), "--config", str(cfg), "--seed", str(seed),
                               "--out", str(out)], cwd=ROOT).returncode
        if code:
            failed.append(out.name)
        print(f"    {'FAILED' if code else 'took'} {time.time() - t:.0f}s", flush=True)
    if failed:
        sys.exit(f"{len(failed)} runs failed: {', '.join(failed)}; rerun to retry them")


if __name__ == "__main__":
    main()
