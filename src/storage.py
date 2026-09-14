"""Exported-storage accounting. Pre-registration §7.

Fixed-width 2-bit ternary encoding plus one fp32 alpha per output filter. BatchNorm ships
separately and is NOT folded, so gamma, beta, running_mean and running_var all count.
No entropy coding, no base-3 packing, no sparse formats (prereg §12).

Storage is therefore a function of architecture alone, computable before training. That is
what makes arm E's selection pre-computable; if a trained model's zero fraction is ever
needed here, something has gone wrong.
"""

import torch.nn as nn

FP32_BITS = 32
TERNARY_BITS = 2
BITS_PER_MB = 8 * 1024 * 1024

# num_batches_tracked is an int64 step counter, not a shipped parameter, so it is excluded.
BN_BUFFERS = ("running_mean", "running_var")


def _bn_buffer_bits(model):
    return sum(
        b.numel() * FP32_BITS
        for name, b in model.named_buffers()
        if name.endswith(BN_BUFFERS)
    )


def ternarizable_conv_names(model):
    """Every conv except the stem (prereg §3). Includes 1x1 downsample convs."""
    return [
        name
        for name, m in model.named_modules()
        if isinstance(m, nn.Conv2d) and name != "conv1"
    ]


def fp32_storage_bits(model):
    """All parameters at fp32, plus BN running stats.

    BN gamma and beta are already inside .parameters(); they are NOT added again here.
    prereg §6 phrases the arm E candidate formula as "parameter count x 32 plus its
    BatchNorm parameters and buffers", which double-counts gamma and beta. Handoff §4.7
    states the correct convention and this follows it. Pending amendment-log entry.
    """
    param_bits = sum(p.numel() for p in model.parameters()) * FP32_BITS
    return param_bits + _bn_buffer_bits(model)


def ternary_storage_bits(model, ternary_conv_names):
    """Ternarized convs at 2 bits/weight + 32 bits per output filter; everything else fp32."""
    ternary = set(ternary_conv_names)
    bits = 0
    for name in ternary:
        m = model.get_submodule(name)
        if not isinstance(m, nn.Conv2d):
            raise TypeError(f"{name} is {type(m).__name__}, not Conv2d")
        bits += m.weight.numel() * TERNARY_BITS + m.weight.shape[0] * FP32_BITS

    quantized = {f"{name}.weight" for name in ternary}
    for pname, p in model.named_parameters():
        if pname not in quantized:
            bits += p.numel() * FP32_BITS

    return bits + _bn_buffer_bits(model)


def n_parameters(model):
    return sum(p.numel() for p in model.parameters())


def bits_per_weight(model, ternary_conv_names=()):
    """Mean bits per parameter under the export convention, for the §7 report table."""
    total = n_parameters(model)
    if not ternary_conv_names:
        return FP32_BITS
    ternary = set(ternary_conv_names)
    quantized = sum(model.get_submodule(n).weight.numel() for n in ternary)
    bits = quantized * TERNARY_BITS + (total - quantized) * FP32_BITS
    bits += sum(model.get_submodule(n).weight.shape[0] for n in ternary) * FP32_BITS
    return bits / total


def mb(bits):
    return bits / BITS_PER_MB
