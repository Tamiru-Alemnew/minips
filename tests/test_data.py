import numpy as np
import torch
from minips import data


def test_shards_are_disjoint_and_cover():
    idx = np.concatenate([data.shard_indices(100, 8, i) for i in range(8)])
    assert sorted(idx.tolist()) == list(range(100))


def test_synthetic_shards_partition_the_data():
    cfg = {"source": "synthetic", "size": 64}
    full = data.load(cfg, train=True)
    parts = [data.load(cfg, train=True, num_shards=4, index=i) for i in range(4)]
    assert sum(len(p) for p in parts) == len(full)
    assert data.shard_len(cfg, 4) == 16


def test_augment_keeps_shape_and_content():
    x = torch.randn(16, 3, 32, 32)
    out = data.augment(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    centre = out[:, :, 4:28, 4:28]
    assert centre.abs().sum() > 0


def test_batches_drop_partial():
    split = data.Split(torch.zeros(10, 4), torch.zeros(10, dtype=torch.long), image=False)
    it = data.batches(split, 4, seed=0)
    assert [next(it)[0].shape[0] for _ in range(4)] == [4, 4, 4, 4]
