"""Quantizer and STE checks. Pre-registration §9."""

import torch

from src.quantizer import THRESHOLD_SCALE, ternarize


def test_three_values_per_filter():
    torch.manual_seed(0)
    w = torch.randn(8, 4, 3, 3)
    w_t, degenerate = ternarize(w)
    assert not degenerate.any()
    for k in range(w.shape[0]):
        vals = torch.unique(w_t[k])
        assert len(vals) <= 3
        alpha = vals.abs().max()
        assert torch.allclose(vals, torch.unique(torch.tensor([-alpha, 0.0, alpha])), atol=1e-6)


def test_alpha_is_mean_magnitude_over_support():
    torch.manual_seed(1)
    w = torch.randn(6, 3, 3, 3)
    w_t, _ = ternarize(w)
    flat, t_flat = w.reshape(6, -1), w_t.reshape(6, -1)
    n = flat.shape[1]
    for k in range(6):
        delta = THRESHOLD_SCALE / n * flat[k].abs().sum()
        support = flat[k].abs() > delta
        expected = flat[k].abs()[support].mean()
        assert torch.allclose(t_flat[k].abs().max(), expected, atol=1e-6)
        # support is exactly where the ternary output is nonzero
        assert torch.equal(t_flat[k] != 0, support)


def test_delta_uses_filter_level_n():
    """Two filters with very different scales: a layer-level Delta would mis-threshold one."""
    w = torch.zeros(2, 1, 2, 2)
    w[0] = torch.tensor([[1.0, 1.0], [1.0, 1.0]]).reshape(1, 2, 2)
    w[1] = torch.tensor([[100.0, 0.0], [0.0, 0.0]]).reshape(1, 2, 2)

    # filter 0: n=4, |w| sum = 4 -> Delta = 0.75. All |w|=1 > 0.75 -> all survive, alpha=1.
    # filter 1: n=4, |w| sum = 100 -> Delta = 18.75. Only the 100 survives, alpha=100.
    w_t, degenerate = ternarize(w)
    assert not degenerate.any()
    assert torch.allclose(w_t[0], torch.ones(1, 2, 2))
    assert torch.allclose(w_t[1].reshape(-1), torch.tensor([100.0, 0.0, 0.0, 0.0]))


def test_degenerate_filter_fallback():
    """A uniform filter has every |w| == Delta exactly, and > is strict, so support is
    empty and the fallback must fire, leaving exactly one surviving weight."""
    n = 4
    w = torch.full((1, 1, 2, 2), 0.75 / n * 4 / 1.0)  # any uniform filter degenerates
    w = torch.full((1, 1, 2, 2), 2.0)
    w_t, degenerate = ternarize(w)
    assert bool(degenerate[0])
    nonzero = w_t.reshape(-1).nonzero().flatten()
    assert len(nonzero) == 1
    assert torch.allclose(w_t.reshape(-1)[nonzero[0]].abs(), torch.tensor(2.0))


def test_degenerate_keeps_sign_and_support_nonempty():
    w = torch.full((3, 1, 2, 2), -5.0)
    w_t, degenerate = ternarize(w)
    assert degenerate.all()
    flat = w_t.reshape(3, -1)
    assert (flat != 0).sum(dim=1).tolist() == [1, 1, 1]  # prereg §9: support never empty
    assert (flat.sum(dim=1) < 0).all()  # sign preserved


def test_ste_is_identity_passthrough():
    """prereg §9: weight.grad must equal the upstream gradient elementwise, unmasked."""
    w = torch.randn(4, 2, 3, 3, requires_grad=True)
    w_t, _ = ternarize(w)
    w_ste = w + (w_t - w).detach()
    upstream = torch.randn_like(w)
    w_ste.backward(upstream)
    assert torch.equal(w.grad, upstream)
