import torch
from minips import data


def test_shards_are_disjoint_and_cover():
    split = data.Split(torch.arange(100).float().view(100, 1), torch.arange(100))
    shards = [data.shard(split, 8, i) for i in range(8)]
    ys = torch.cat([s.y for s in shards])
    assert sorted(ys.tolist()) == list(range(100))


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
