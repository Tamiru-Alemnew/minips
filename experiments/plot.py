"""Regenerates every README figure and table from the JSON files in results/<matrix>/."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASELINE = "bsp_none"
MODES = ["bsp", "ssp2", "ssp4", "asp"]
COMPRESSION = ["none", "topk", "int8"]
COLORS = {"bsp": "#2a6fdb", "ssp2": "#14a37f", "ssp4": "#e08a00", "asp": "#c0392b", "allreduce": "#555555"}


def load(matrix: str) -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for f in sorted((ROOT / "results" / matrix).glob("*_s*.json")):
        r = json.loads(f.read_text())
        runs[r["name"]].append(r)
    return runs


def med(rs: list[dict], key: str) -> float:
    return statistics.median(r[key] for r in rs)


def spread(rs: list[dict], key: str) -> str:
    vals = [r[key] for r in rs]
    return "" if len(vals) < 2 else f" ({min(vals):.3g}–{max(vals):.3g})"


def summary(runs: dict[str, list[dict]]) -> list[dict]:
    base = runs.get(BASELINE)
    rows = []
    for name, rs in sorted(runs.items()):
        row = {
            "name": name, "seeds": len(rs), "acc": med(rs, "final_acc"), "wall": med(rs, "wallclock_s"),
            "bytes": med(rs, "bytes_total"), "wait": statistics.median(r.get("mean_wait_s", 0.0) for r in rs),
            "acc_spread": spread(rs, "final_acc"), "system": rs[0]["system"],
        }
        if base:
            row["comm_x"] = med(base, "bytes_total") / row["bytes"]
            row["speedup"] = med(base, "wallclock_s") / row["wall"]
            row["acc_delta"] = row["acc"] - med(base, "final_acc")
        rows.append(row)
    return rows


def table(rows: list[dict]) -> str:
    out = [
        "| run | seeds | test acc | Δ acc vs BSP dense | wall-clock (s) | speed-up | bytes on the wire (MiB) "
        "| comm. reduction | mean wait / worker (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ps = r["system"] == "ps"
        cells = [
            r["name"], str(r["seeds"]), f"{r['acc']:.4f}{r['acc_spread']}", f"{r.get('acc_delta', 0) * 100:+.2f} pp",
            f"{r['wall']:.0f}", f"{r.get('speedup', 1):.2f}×", f"{r['bytes'] / 2**20:,.0f}" + ("" if ps else "*"),
            f"{r.get('comm_x', 1):.1f}×", f"{r['wait']:.1f}" if ps else "n/a",
        ]
        out.append("| " + " | ".join(cells) + " |")
    out.append("\n\\* All-reduce bytes are computed from DDP bucket sizes (ring: 2(N−1)/N per rank), not measured.")
    return "\n".join(out) + "\n"


def plot_curves(runs, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in ["bsp_none", "bsp_topk", "ssp4_topk", "asp_none", "allreduce"]:
        if name not in runs:
            continue
        r = runs[name][0]
        key = "allreduce" if name == "allreduce" else name.split("_")[0]
        style = "--" if "topk" in name else "-"
        ax.plot([p["time"] for p in r["curve"]], [p["acc"] for p in r["curve"]], style, marker="o", ms=3,
                color=COLORS[key], label=name)
    ax.set_xlabel("wall-clock (s)")
    ax.set_ylabel("test accuracy (curve subset)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_time.png", dpi=150)
    plt.close(fig)


def plot_bytes(runs, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    width = 0.2
    for i, mode in enumerate(MODES):
        vals = []
        for comp in COMPRESSION:
            rs = runs.get(f"{mode}_{comp}")
            epochs = rs[0]["config"]["epochs"] if rs else 1
            vals.append(med(rs, "bytes_total") / epochs / 2**20 if rs else 0)
        ax.bar([x + width * i for x in range(len(COMPRESSION))], vals, width, color=COLORS[mode], label=mode)
    ax.set_xticks([x + 1.5 * width for x in range(len(COMPRESSION))], COMPRESSION)
    ax.set_ylabel("MiB on the wire per epoch")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out / "bytes_per_epoch.png", dpi=150)
    plt.close(fig)


def plot_stragglers(runs, out: Path):
    names = ["allreduce", "bsp_none", "bsp_topk", "ssp2_topk", "ssp4_topk"]
    names = [n for n in names if n in runs and f"{n}_straggler" in runs]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], [med(runs[n], "wallclock_s") for n in names], 0.4, color="#9db7e8",
           label="all workers equal")
    ax.bar([i + 0.2 for i in x], [med(runs[f"{n}_straggler"], "wallclock_s") for n in names], 0.4,
           color="#2a6fdb", label="one worker 50% slower")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("wall-clock (s)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "stragglers.png", dpi=150)
    plt.close(fig)


def plot_staleness(runs, out: Path):
    names = [n for n in ["ssp2_topk", "ssp4_topk", "asp_topk", "ssp4_topk_straggler"] if n in runs]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for n in names:
        hist = {int(k): v for k, v in runs[n][0]["staleness_hist"].items()}
        total = sum(hist.values())
        ks = sorted(hist)
        ax.plot(ks, [hist[k] / total for k in ks], marker="o", ms=3, label=n)
    ax.set_xlabel("clock gap to the slowest worker at read time")
    ax.set_ylabel("share of reads")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "staleness.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="full")
    args = ap.parse_args()
    runs = load(args.matrix)
    if not runs:
        raise SystemExit(f"no results in results/{args.matrix}/")
    out = ROOT / "results" / args.matrix
    md = table(summary(runs))
    (out / "table.md").write_text(md)
    plot_curves(runs, out)
    plot_bytes(runs, out)
    plot_stragglers(runs, out)
    plot_staleness(runs, out)
    print(md)


if __name__ == "__main__":
    main()
