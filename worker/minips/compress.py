import math
import struct

import numpy as np
import torch

from .pb import ps_pb2

DENSE, TOPK, INT8 = ps_pb2.DENSE_F32, ps_pb2.TOPK, ps_pb2.INT8


class Dense:
    def encode(self, name: str, grad: torch.Tensor) -> tuple[int, bytes]:
        return DENSE, grad.detach().reshape(-1).numpy().astype("<f4").tobytes()


class TopK:
    """Top-k sparsification with error feedback.

    Whatever is not sent (the unsent entries plus the float16 rounding error
    of the sent ones) is kept in a residual and added to the next gradient.
    """

    def __init__(self, ratio: float, error_feedback: bool = True):
        self.ratio = ratio
        self.error_feedback = error_feedback
        self.residual: dict[str, torch.Tensor] = {}

    def encode(self, name: str, grad: torch.Tensor) -> tuple[int, bytes]:
        acc = grad.detach().reshape(-1).float()
        if self.error_feedback and name in self.residual:
            acc = acc + self.residual[name]
        k = max(1, math.ceil(self.ratio * acc.numel()))
        idx = torch.topk(acc.abs(), k, sorted=False).indices
        vals = acc[idx].half()
        if self.error_feedback:
            res = acc.clone()
            res[idx] -= vals.float()
            self.residual[name] = res
        idx_np = idx.numpy().astype("<u4")
        return TOPK, struct.pack("<I", k) + idx_np.tobytes() + vals.numpy().astype("<f2").tobytes()


class Int8:
    def encode(self, name: str, grad: torch.Tensor) -> tuple[int, bytes]:
        g = grad.detach().reshape(-1).float().numpy()
        peak = float(np.abs(g).max()) if g.size else 0.0
        scale = peak / 127 if peak > 0 else 1.0
        q = np.clip(np.rint(g / scale), -127, 127).astype(np.int8)
        return INT8, struct.pack("<f", scale) + q.tobytes()


def make(cfg: dict | None):
    cfg = cfg or {"kind": "none"}
    kind = cfg.get("kind", "none")
    if kind == "none":
        return Dense()
    if kind == "topk":
        return TopK(cfg.get("ratio", 0.01), cfg.get("error_feedback", True))
    if kind == "int8":
        return Int8()
    raise ValueError(f"unknown compression {kind!r}")


def decode(encoding: int, data: bytes, numel: int) -> np.ndarray:
    if encoding == DENSE:
        return np.frombuffer(data, "<f4").copy()
    if encoding == TOPK:
        (k,) = struct.unpack_from("<I", data)
        idx = np.frombuffer(data, "<u4", k, 4)
        vals = np.frombuffer(data, "<f2", k, 4 + 4 * k).astype(np.float32)
        out = np.zeros(numel, np.float32)
        out[idx] = vals
        return out
    if encoding == INT8:
        (scale,) = struct.unpack_from("<f", data)
        return np.frombuffer(data, np.int8, numel, 4).astype(np.float32) * scale
    raise ValueError(f"unknown encoding {encoding}")
