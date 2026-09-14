"""TWN closed-form ternarization, per output filter. Pre-registration §4.

Recomputed on every forward pass from the current latent weights -- never cached per
epoch. Nothing here is learned: alpha and Delta are closed-form functions of the latent
weights, per TWN Algorithm 1.
"""

import torch

THRESHOLD_SCALE = 0.75  # TWN Algorithm 1 (arXiv:1605.04711v3), per-filter


def ternarize(w):
    """Ternarize a conv weight per output filter.

    Args:
        w: (out_ch, in_ch, kh, kw) latent weights.

    Returns:
        (w_ternary, degenerate) where w_ternary holds values in {-alpha_k, 0, +alpha_k}
        for filter k, and degenerate is a (out_ch,) bool mask, True where the empty-support
        fallback fired.
    """
    out_ch = w.shape[0]
    w_flat = w.reshape(out_ch, -1)
    n = w_flat.shape[1]  # filter-level n, NOT layer-level (prereg §4)
    w_abs = w_flat.abs()

    delta = (THRESHOLD_SCALE / n) * w_abs.sum(dim=1, keepdim=True)
    mask = w_abs > delta  # strict >, TWN Eq. 3

    # Degenerate filters (empty support): force the largest-magnitude weight into the
    # support with its own sign. Written branch-free rather than as `if degenerate.any()`
    # because that test is a GPU->host sync, and this runs once per conv per step
    # (handoff §12: no host syncs in the forward path). Result is identical.
    degenerate = ~mask.any(dim=1)
    largest = w_abs.argmax(dim=1, keepdim=True)
    forced = torch.zeros_like(mask).scatter_(1, largest, True) & degenerate.unsqueeze(1)
    mask = mask | forced

    counts = mask.sum(dim=1, keepdim=True)  # >0 everywhere; asserted in tests, not here
    signs = w_flat.sign() * mask
    alpha = (w_abs * mask).sum(dim=1, keepdim=True) / counts

    return (alpha * signs).reshape(w.shape), degenerate
