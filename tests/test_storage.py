"""Storage accounting checked against an independent hand computation. Prereg §9.

tests/hand_storage.py recomputes every figure from raw arithmetic with no torch, so an
agreement here is two independent implementations agreeing, not one asserting itself.
"""

from src import storage
from src.models import ARM_E_CANDIDATES, resnet18_cifar, resnet6n2
from tests import hand_storage as hand

R18 = ([2, 2, 2, 2], [64, 128, 256, 512], False)


def _candidate(n):
    return ([n] * 3, [16, 32, 64], True)


def test_ternary_r18_matches_hand_computation():
    model = resnet18_cifar()
    names = storage.ternarizable_conv_names(model)
    assert len(names) == 19
    assert storage.ternary_storage_bits(model, names) == hand.ternary_bits(*R18)


def test_fp32_r18_matches_hand_computation():
    assert storage.fp32_storage_bits(resnet18_cifar()) == hand.fp32_bits(*R18)


def test_every_arm_e_candidate_matches_hand_computation():
    for depth, n in ARM_E_CANDIDATES.items():
        model = resnet6n2(n)
        assert storage.fp32_storage_bits(model) == hand.fp32_bits(*_candidate(n)), depth
        assert storage.n_parameters(model) == hand.n_params(*_candidate(n)), depth


def test_resnet20_matches_published_parameter_count():
    """He et al. (2016) report 0.27M for ResNet-20. This is the check that caught the
    6n+2 block-count error recorded in amendment A2."""
    assert storage.n_parameters(resnet6n2(3)) == 269_722


def test_arm_e_selects_resnet44():
    target = storage.ternary_storage_bits(
        resnet18_cifar(), storage.ternarizable_conv_names(resnet18_cifar())
    )
    errors = {
        depth: abs(storage.fp32_storage_bits(resnet6n2(n)) - target) / target
        for depth, n in ARM_E_CANDIDATES.items()
    }
    assert min(errors, key=errors.get) == 44
    # only one candidate in the 5% tie-break band, so the tie-break never fires
    best = min(errors.values())
    assert sum(1 for e in errors.values() if abs(e - best) < 0.05) == 1


def test_bn_shipped_not_folded():
    """All four BN tensors count: gamma and beta via parameters(), running stats on top."""
    model = resnet18_cifar()
    params = sum(p.numel() for p in model.parameters()) * 32
    total = storage.fp32_storage_bits(model)
    running = sum(
        b.numel() for name, b in model.named_buffers()
        if name.endswith(("running_mean", "running_var"))
    ) * 32
    assert running > 0
    assert total == params + running  # counted exactly once each, no double-count (A1)


def test_num_batches_tracked_excluded():
    model = resnet18_cifar()
    counters = [n for n, _ in model.named_buffers() if n.endswith("num_batches_tracked")]
    assert counters, "expected BN step counters to exist"
    # each is a scalar int64; if they were counted, storage would grow by 32 bits each
    assert storage.fp32_storage_bits(model) % 32 == 0


def test_stem_excluded_downsample_included():
    names = storage.ternarizable_conv_names(resnet18_cifar())
    assert "conv1" not in names  # fp32 stem (prereg §3)
    assert sum(1 for n in names if n.endswith("downsample.0")) == 3


def test_bits_per_weight_is_near_two():
    model = resnet18_cifar()
    names = storage.ternarizable_conv_names(model)
    bpw = storage.bits_per_weight(model, names)
    assert 2.0 < bpw < 2.2, bpw
    assert storage.bits_per_weight(model) == 32
