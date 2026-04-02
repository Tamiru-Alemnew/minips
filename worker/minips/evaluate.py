import torch
from torch import nn

from . import data


@torch.no_grad()
def recalibrate_bn(model: nn.Module, split: data.Split, batches: int, batch_size: int = 256) -> None:
    """Recompute BatchNorm running statistics for the given weights.

    Workers keep BN statistics locally (they are not gradients), so a model
    assembled from server weights needs fresh statistics before evaluation.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(split), generator=g)
    for i in range(min(batches, len(split) // batch_size)):
        model(split.x[perm[i * batch_size : (i + 1) * batch_size]])


@torch.no_grad()
def accuracy(model: nn.Module, split: data.Split, batch_size: int = 500) -> tuple[float, float]:
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(split), batch_size):
        x, y = split.x[i : i + batch_size], split.y[i : i + batch_size]
        out = model(x)
        loss += nn.functional.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
    return correct / len(split), loss / len(split)


def subset(split: data.Split, size: int) -> data.Split:
    """A fixed random subset; test files can be sorted by class, so never take a prefix."""
    idx = torch.randperm(len(split), generator=torch.Generator().manual_seed(0))[:size]
    return data.Split(split.x[idx], split.y[idx], split.image)


def final(model: nn.Module, train: data.Split, test: data.Split, bn_batches: int = 50) -> tuple[float, float]:
    recalibrate_bn(model, train, bn_batches)
    return accuracy(model, test)
