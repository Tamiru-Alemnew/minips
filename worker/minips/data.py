from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


@dataclass
class Split:
    x: torch.Tensor
    y: torch.Tensor
    image: bool = True

    def __len__(self):
        return len(self.y)


def _to_split(x_uint8_nhwc: np.ndarray, y: np.ndarray) -> Split:
    x = torch.from_numpy(x_uint8_nhwc).permute(0, 3, 1, 2).float().div_(255)
    return Split((x - MEAN) / STD, torch.from_numpy(y.astype(np.int64)))


def _torchvision(root: str, train: bool) -> tuple[np.ndarray, np.ndarray]:
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root, train=train, download=True)
    return ds.data, np.array(ds.targets)


def _folder(root: str, train: bool) -> tuple[np.ndarray, np.ndarray]:
    """CIFAR-10 stored as <root>/{train,test}/<class>/*.jpg, cached to .npy on first use."""
    part = "train" if train else "test"
    xf, yf = Path(root) / f"{part}_x.npy", Path(root) / f"{part}_y.npy"
    if not yf.exists():
        from PIL import Image

        xs, ys = [], []
        for label, cls in enumerate(CIFAR_CLASSES):
            for f in sorted((Path(root) / part / cls).glob("*")):
                xs.append(np.asarray(Image.open(f).convert("RGB")))
                ys.append(label)
        np.save(xf, np.stack(xs))
        np.save(yf, np.array(ys))
    return np.load(xf, mmap_mode="r"), np.load(yf)


def _synthetic(train: bool, n: int, dim: int, classes: int) -> Split:
    centers = torch.randn(classes, dim, generator=torch.Generator().manual_seed(0)) * 0.6
    g = torch.Generator().manual_seed(1 if train else 2)
    y = torch.randint(classes, (n,), generator=g)
    x = centers[y] + torch.randn(n, dim, generator=g)
    return Split(x, y, image=False)


def shard_indices(size: int, num_shards: int, index: int) -> np.ndarray:
    """Disjoint shards from one fixed permutation, identical for every run."""
    if num_shards == 1:
        return np.arange(size)
    perm = np.random.default_rng(0).permutation(size)
    return np.array_split(perm, num_shards)[index]


def load(cfg: dict, train: bool, num_shards: int = 1, index: int = 0) -> Split:
    """Loads one shard of a split; only that shard is ever converted to float."""
    source = cfg.get("source", "torchvision")
    if source == "synthetic":
        split = _synthetic(train, cfg.get("size", 2048), cfg.get("dim", 32), cfg.get("classes", 4))
        idx = torch.from_numpy(shard_indices(len(split), num_shards, index))
        return Split(split.x[idx], split.y[idx], image=False)
    if source == "torchvision":
        x, y = _torchvision(cfg.get("root", "data"), train)
    elif source == "folder":
        x, y = _folder(cfg["root"], train)
    else:
        raise ValueError(f"unknown data source {source!r}")
    keep = np.arange(len(y))
    limit = cfg.get("train_size" if train else "test_size")
    if limit:
        keep = np.sort(np.random.default_rng(0).permutation(len(y))[:limit])
    keep = keep[shard_indices(len(keep), num_shards, index)]
    return _to_split(np.ascontiguousarray(x[keep]), y[keep])


def shard_len(cfg: dict, num_shards: int) -> int:
    """Size of the smallest training shard, which sets the steps per epoch."""
    size = cfg.get("train_size") or len(load(cfg, train=True).y)
    return min(len(a) for a in np.array_split(np.arange(size), num_shards))


def augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Random 32x32 crop from a 4-pixel zero-padded image, plus horizontal flip."""
    n = x.shape[0]
    padded = torch.nn.functional.pad(x, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (n,), generator=g)
    dy = torch.randint(0, 9, (n,), generator=g)
    rows = (dy.view(n, 1) + torch.arange(32)).view(n, 1, 32, 1).expand(n, x.shape[1], 32, 40)
    out = padded.gather(2, rows)
    cols = (dx.view(n, 1) + torch.arange(32)).view(n, 1, 1, 32).expand(n, x.shape[1], 32, 32)
    out = out.gather(3, cols)
    flip = torch.rand(n, generator=g) < 0.5
    out[flip] = out[flip].flip(3)
    return out


def batches(split: Split, batch_size: int, seed: int, augment_images: bool = True):
    """Endless stream of shuffled batches; drops the last partial batch of each epoch."""
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = torch.randperm(len(split), generator=g)
        for i in range(0, len(split) - batch_size + 1, batch_size):
            idx = perm[i : i + batch_size]
            x = split.x[idx]
            if split.image and augment_images:
                x = augment(x, g)
            yield x, split.y[idx]
