"""KD loss checks. Pre-registration §9."""

import torch
import torch.nn.functional as F

from src.losses import kd_loss


def test_lambda_zero_is_plain_cross_entropy():
    torch.manual_seed(0)
    zs, zt = torch.randn(16, 10), torch.randn(16, 10)
    y = torch.randint(0, 10, (16,))
    assert torch.allclose(kd_loss(zs, zt, y, lam=0.0), F.cross_entropy(zs, y), atol=1e-7)


def test_identical_logits_give_zero_soft_term():
    torch.manual_seed(1)
    z = torch.randn(8, 10)
    y = torch.randint(0, 10, (8,))
    # lam=1 isolates the soft term; teacher == student -> KL is 0.
    assert torch.allclose(kd_loss(z, z.clone(), y, lam=1.0), torch.tensor(0.0), atol=1e-6)


def _soft_grad_norm(T):
    """Gradient norm of the T^2-scaled soft term alone, w.r.t. the student logits."""
    torch.manual_seed(2)
    zs = torch.randn(32, 10, requires_grad=True)
    zt = torch.randn(32, 10)
    y = torch.zeros(32, dtype=torch.long)
    kd_loss(zs, zt, y, T=T, lam=1.0).backward()
    return zs.grad.norm().item()


def test_t_squared_factor_is_in_the_right_place():
    """The T^2 factor compensates the 1/T^2 shrinkage of the softened-softmax gradient.

    prereg §9 asks for "approximately equal at T=1 and T=4", but the compensation is a
    high-temperature limit and T=1 is outside it, so there is no honest tolerance at which
    those two are equal. What the check must actually discriminate is a MISPLACED factor,
    which shows up as a ~T^2 (16x) gap. So: assert the T=1 vs T=4 ratio stays well under
    that, and assert near-equality at T=4 vs T=8, where the limit does hold.
    """
    g1, g4, g8 = _soft_grad_norm(1.0), _soft_grad_norm(4.0), _soft_grad_norm(8.0)

    assert 0.25 < g4 / g1 < 4.0, f"T=1 vs T=4 ratio {g4 / g1:.2f} looks like a misplaced T^2"
    assert 0.8 < g8 / g4 < 1.25, f"T=4 vs T=8 ratio {g8 / g4:.3f} should be ~1"


def test_batchmean_not_mean():
    """'mean' divides by the class count as well, scaling the soft term by 1/C.

    Constructed so the true KL is unchanged: classes 10..19 carry essentially no mass in
    either distribution, so widening from C=10 to C=20 leaves the distributions alone.
    batchmean is invariant; 'mean' would halve.
    """
    torch.manual_seed(3)
    zs10, zt10 = torch.randn(16, 10), torch.randn(16, 10)
    y = torch.randint(0, 10, (16,))

    pad = torch.full((16, 10), -1e4)
    zs20 = torch.cat([zs10, pad], dim=1)
    zt20 = torch.cat([zt10, pad], dim=1)

    loss10 = kd_loss(zs10, zt10, y, lam=1.0)
    loss20 = kd_loss(zs20, zt20, y, lam=1.0)
    assert torch.allclose(loss10, loss20, atol=1e-5), (
        f"{loss10.item():.6f} != {loss20.item():.6f}; reduction is probably 'mean'"
    )


def test_soft_to_hard_ratio_is_nine_to_one():
    """lam/(1-lam) = 9 at lam=0.9: the hard term's weight is 0.1 of the CE gradient."""
    torch.manual_seed(4)
    zs = torch.randn(16, 10, requires_grad=True)
    y = torch.randint(0, 10, (16,))
    kd_loss(zs, zs.detach().clone(), y, lam=0.9).backward()

    # teacher == student zeroes the soft gradient, leaving exactly (1-lam) * dCE.
    zs2 = zs.detach().clone().requires_grad_()
    F.cross_entropy(zs2, y).backward()
    assert torch.allclose(zs.grad, 0.1 * zs2.grad, atol=1e-6)
