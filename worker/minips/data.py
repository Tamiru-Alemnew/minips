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


def _torchvision(root: str, train: bool) -> Split:
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root, train=train, download=True)
    return _to_split(ds.data, np.array(ds.targets))


def _folder(root: str, train: bool) -> Split:
    """CIFAR-10 stored as <root>/{train,test}/<class>/*.jpg, cached to .npz on first use."""
    part = "train" if train else "test"
    cache = Path(root) / f"{part}.npz"
    if not cache.exists():
        from PIL import Image

        xs, ys = [], []
        for label, cls in enumerate(CIFAR_CLASSES):
            for f in sorted((Path(root) / part / cls).glob("*")):
                xs.append(np.asarray(Image.open(f).convert("RGB")))
                ys.append(label)
        np.savez(cache, x=np.stack(xs), y=np.array(ys))
    d = np.load(cache)
    return _to_split(d["x"], d["y"])


def _synthetic(train: bool, n: int, dim: int, classes: int) -> Split:
    centers = torch.randn(classes, dim, generator=torch.Generator().manual_seed(0)) * 0.6
    g = torch.Generator().manual_seed(1 if train else 2)
    y = torch.randint(classes, (n,), generator=g)
    x = centers[y] + torch.randn(n, dim, generator=g)
    return Split(x, y, image=False)


def load(cfg: dict, train: bool) -> Split:
    source = cfg.get("source", "torchvision")
    if source == "torchvision":
        split = _torchvision(cfg.get("root", "data"), train)
    elif source == "folder":
        split = _folder(cfg["root"], train)
    elif source == "synthetic":
        split = _synthetic(train, cfg.get("size", 2048), cfg.get("dim", 32), cfg.get("classes", 4))
    else:
        raise ValueError(f"unknown data source {source!r}")
    limit = cfg.get("train_size" if train else "test_size")
    if limit:
        idx = torch.randperm(len(split), generator=torch.Generator().manual_seed(0))[:limit]
        split = Split(split.x[idx], split.y[idx], split.image)
    return split


def shard(split: Split, num_shards: int, index: int) -> Split:
    """Disjoint shards from one fixed permutation, identical for every run."""
    perm = torch.randperm(len(split), generator=torch.Generator().manual_seed(0))
    idx = torch.tensor_split(perm, num_shards)[index]
    return Split(split.x[idx], split.y[idx], split.image)


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
