import numpy as np
import pytest
import torch
from minips import compress


def grad(n=1000, seed=0):
    return torch.randn(n, generator=torch.Generator().manual_seed(seed))


def test_dense_roundtrip():
    g = grad()
    enc, data = compress.Dense().encode("w", g)
    np.testing.assert_array_equal(compress.decode(enc, data, g.numel()), g.numpy())


def test_topk_sends_k_largest():
    g = grad()
    enc, data = compress.TopK(0.01).encode("w", g)
    out = compress.decode(enc, data, g.numel())
    assert np.count_nonzero(out) == 10
    assert len(data) == 4 + 10 * 6
    top = np.argsort(-np.abs(g.numpy()))[:10]
    assert set(np.flatnonzero(out)) == set(top)


def test_topk_residual_sums_back_to_gradient():
    g = grad()
    c = compress.TopK(0.05)
    enc, data = c.encode("w", g)
    sent = compress.decode(enc, data, g.numel())
    np.testing.assert_allclose(sent + c.residual["w"].numpy(), g.numpy(), atol=1e-6)


def test_topk_error_feedback_delivers_everything_eventually():
    g = grad(200)
    c = compress.TopK(0.1)
    total = np.zeros(200, np.float32)
    for step in range(200):
        enc, data = c.encode("w", g if step == 0 else torch.zeros(200))
        total += compress.decode(enc, data, 200)
    np.testing.assert_allclose(total, g.numpy(), atol=1e-3)


def test_topk_without_feedback_keeps_no_state():
    c = compress.TopK(0.01, error_feedback=False)
    c.encode("w", grad())
    assert c.residual == {}


def test_momentum_correction_masks_sent_entries():
    c = compress.TopK(0.1, momentum=0.9)
    g = grad(100)
    enc, data = c.encode("w", g)
    sent = np.flatnonzero(compress.decode(enc, data, 100))
    assert np.all(c.velocity["w"].numpy()[sent] == 0)
    unsent = np.setdiff1d(np.arange(100), sent)
    np.testing.assert_allclose(c.velocity["w"].numpy()[unsent], g.numpy()[unsent])
    enc, data = c.encode("w", torch.zeros(100))
    np.testing.assert_allclose(c.velocity["w"].numpy()[unsent][:5] / g.numpy()[unsent][:5], 0.9, rtol=1e-5)


def test_int8_error_within_half_scale():
    g = grad(5000) * 3
    enc, data = compress.Int8().encode("w", g)
    out = compress.decode(enc, data, g.numel())
    scale = np.abs(g.numpy()).max() / 127
    assert np.abs(out - g.numpy()).max() <= scale / 2 + 1e-6
    assert len(data) == 4 + 5000


def test_int8_zero_gradient():
    enc, data = compress.Int8().encode("w", torch.zeros(8))
    np.testing.assert_array_equal(compress.decode(enc, data, 8), np.zeros(8))


def test_unknown_kind():
    with pytest.raises(ValueError):
        compress.make({"kind": "zip"})
