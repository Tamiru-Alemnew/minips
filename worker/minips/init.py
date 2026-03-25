import argparse
import json
import struct

import torch

from . import config, models

MAGIC = b"MPS1"


def write(model: torch.nn.Module, path: str) -> None:
    named = list(model.named_parameters())
    header = json.dumps(
        [{"name": n, "shape": list(p.shape), "dtype": "float32"} for n, p in named]
    ).encode()
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(header)) + header)
        for _, p in named:
            f.write(p.detach().float().contiguous().numpy().astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser(description="Write initial weights for the parameter server.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = config.load(args.config)
    torch.manual_seed(args.seed)
    write(models.build(cfg["model"]), args.out)


if __name__ == "__main__":
    main()
