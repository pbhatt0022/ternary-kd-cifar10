"""Model construction and ternary surgery. Pre-registration §3."""

import torch
import torch.nn as nn

from src.models import EXPECTED_TERNARY_CONVS, resnet18_cifar, resnet34_cifar, resnet6n2
from src.ternary_modules import TernaryConv2d, ternarize_model


def test_cifar_stem_and_no_maxpool():
    model = resnet18_cifar()
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert model.conv1.padding == (1, 1)
    assert not any(isinstance(m, nn.MaxPool2d) for m in model.modules())


def test_bn_initialised_to_one_and_zero_everywhere():
    """prereg §12: NO zero-gamma init of the final BN in each residual block."""
    for model in (resnet18_cifar(), resnet6n2(3)):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert torch.all(m.weight == 1.0)
                assert torch.all(m.bias == 0.0)


def test_ternarize_model_swaps_expected_count():
    model = resnet18_cifar()
    names = ternarize_model(model, expected_count=EXPECTED_TERNARY_CONVS["resnet18_cifar"])
    assert len(names) == 19
    assert isinstance(model.conv1, nn.Conv2d) and not isinstance(model.conv1, TernaryConv2d)
    swapped = [n for n, m in model.named_modules() if isinstance(m, TernaryConv2d)]
    assert sorted(swapped) == sorted(names)


def test_wrong_expected_count_raises():
    try:
        ternarize_model(resnet18_cifar(), expected_count=18)
    except RuntimeError:
        return
    raise AssertionError("expected a RuntimeError on a count mismatch")


def test_swap_preserves_weights_bit_exactly():
    """A ternary arm and an fp arm at the same seed must start from identical weights."""
    torch.manual_seed(7)
    fp = resnet18_cifar()
    torch.manual_seed(7)
    tern = resnet18_cifar()
    ternarize_model(tern)

    for (n1, p1), (n2, p2) in zip(fp.named_parameters(), tern.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), n1


def test_forward_shapes():
    x = torch.randn(2, 3, 32, 32)
    for model in (resnet18_cifar(), resnet34_cifar(), resnet6n2(3)):
        assert model(x).shape == (2, 10)


def test_ternary_forward_records_degenerate_and_quantizes():
    model = resnet18_cifar()
    ternarize_model(model)
    model(torch.randn(2, 3, 32, 32))
    conv = model.get_submodule("layer1.0.conv1")
    assert hasattr(conv, "_last_degenerate")
    assert conv._last_degenerate.shape == (conv.out_channels,)
    # the forward must have used ternary values, not the latent weights
    from src.quantizer import ternarize
    w_t, _ = ternarize(conv.weight)
    assert len(torch.unique(w_t[0])) <= 3


def test_6n2_uses_parameter_free_shortcuts():
    model = resnet6n2(3)
    convs = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    assert not any("downsample" in n for n in convs)  # option A: no 1x1 projections
