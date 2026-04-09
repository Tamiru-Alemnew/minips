"""Writes experiments/configs/<matrix>/*.yaml: one file per run in the results table."""
from pathlib import Path

import yaml

HERE = Path(__file__).parent / "configs"

BASE = {
    "full": {
        "workers": 8, "epochs": 30, "batch_size": 128, "lr": 0.4, "warmup_epochs": 2,
        "momentum": 0.9, "weight_decay": 5e-4,
        "model": {"name": "resnet18", "width": 64},
        "data": {"source": "torchvision", "root": "data"},
        "threads_per_worker": 1, "bn_batches": 50, "curve_test_size": 2000,
    },
    "smoke": {
        "workers": 4, "epochs": 8, "batch_size": 128, "lr": 0.4, "warmup_epochs": 1,
        "momentum": 0.9, "weight_decay": 5e-4,
        "model": {"name": "resnet18", "width": 16},
        "data": {"source": "folder", "root": "data/cifar10-jpg", "train_size": 20000},
        "threads_per_worker": 1, "bn_batches": 20, "curve_test_size": 2000,
    },
}

MODES = {
    "bsp": {"mode": "bsp"},
    "ssp2": {"mode": "ssp", "staleness": 2},
    "ssp4": {"mode": "ssp", "staleness": 4},
    "asp": {"mode": "asp"},
}
COMPRESSION = {
    "none": {"kind": "none"},
    "topk": {"kind": "topk", "ratio": 0.01},
    "int8": {"kind": "int8"},
}
STRAGGLER = {"worker": 1, "slowdown": 0.5}
STRAGGLER_RUNS = ["bsp_none", "bsp_topk", "ssp2_topk", "ssp4_topk"]


def runs():
    for m, ps in MODES.items():
        for c, comp in COMPRESSION.items():
            yield f"{m}_{c}", {"system": "ps", "ps": {**ps, "shards": 4}, "compression": comp}
    yield "bsp_topk_noef", {"system": "ps", "ps": {"mode": "bsp", "shards": 4},
                            "compression": {**COMPRESSION["topk"], "error_feedback": False}}
    for m in ("bsp", "ssp4"):
        yield f"{m}_topk_mc", {"system": "ps", "ps": {**MODES[m], "shards": 4}, "momentum": 0.0,
                               "compression": {**COMPRESSION["topk"], "momentum_correction": 0.9}}
    yield "allreduce", {"system": "allreduce"}
    for name in STRAGGLER_RUNS:
        m, c = name.split("_")
        yield f"{name}_straggler", {"system": "ps", "ps": {**MODES[m], "shards": 4},
                                    "compression": COMPRESSION[c], "straggler": STRAGGLER}
    yield "allreduce_straggler", {"system": "allreduce", "straggler": STRAGGLER}


def main():
    for matrix, base in BASE.items():
        out = HERE / matrix
        out.mkdir(parents=True, exist_ok=True)
        for name, extra in runs():
            cfg = {"name": name, **base, **extra}
            (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main()
