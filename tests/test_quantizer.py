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


def test_degenerate_fires_only_on_an_all_zero_filter():
    """Delta = 0.75 * mean|w|, and max|w| >= mean|w| > 0.75 * mean|w| for any filter that
    is not identically zero. So the support is non-empty for every nonzero filter, and the
    fallback can fire only on an exactly-zero one -- which is the absorbing state the
    pre-registration introduces it to rescue."""
    w = torch.zeros(3, 1, 2, 2)
    w[1] = 1.0  # uniform: Delta = 0.75, every weight survives
    w[2] = torch.tensor([0.0, 0.0, 0.0, 7.0]).reshape(1, 2, 2)  # spike: only the 7 survives
    _, degenerate = ternarize(w)
    assert degenerate.tolist() == [True, False, False]


def test_uniform_filter_keeps_every_weight():
    """Delta = 0.75 * c for a filter uniformly equal to c, and c > 0.75 * c, so nothing is
    thresholded away and alpha == c."""
    w = torch.full((1, 1, 2, 2), 3.0)
    w_t, degenerate = ternarize(w)
    assert not degenerate.any()
    assert torch.allclose(w_t, torch.full((1, 1, 2, 2), 3.0))


def test_all_zero_filter_is_finite_not_nan():
    """What the fallback actually buys: with an empty support the divisor would be 0 and
    alpha would be 0/0 = NaN, poisoning the forward pass."""
    w = torch.zeros(2, 1, 3, 3)
    w_t, degenerate = ternarize(w)
    assert degenerate.all()
    assert torch.isfinite(w_t).all()
    assert torch.all(w_t == 0)


def test_degenerate_never_fires_on_random_weights():
    torch.manual_seed(5)
    for scale in (1e-4, 1.0, 1e3):
        _, degenerate = ternarize(torch.randn(16, 8, 3, 3) * scale)
        assert not degenerate.any()


def test_support_is_never_empty():
    """prereg section 9: assert the support is non-empty after the fallback."""
    torch.manual_seed(6)
    for w in (torch.randn(8, 4, 3, 3), torch.zeros(4, 2, 3, 3), torch.full((3, 1, 2, 2), -5.0)):
        w_t, _ = ternarize(w)
        counts = (w_t.reshape(w.shape[0], -1) != 0).sum(dim=1)
        # an all-zero filter has a one-element support whose value is itself zero, so
        # count nonzeros only where the filter had any magnitude to begin with
        had_magnitude = w.reshape(w.shape[0], -1).abs().sum(dim=1) > 0
        assert (counts[had_magnitude] > 0).all()


def test_ste_is_identity_passthrough():
    """prereg §9: weight.grad must equal the upstream gradient elementwise, unmasked."""
    w = torch.randn(4, 2, 3, 3, requires_grad=True)
    w_t, _ = ternarize(w)
    w_ste = w + (w_t - w).detach()
    upstream = torch.randn_like(w)
    w_ste.backward(upstream)
    assert torch.equal(w.grad, upstream)
