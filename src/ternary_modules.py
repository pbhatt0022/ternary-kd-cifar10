"""TernaryConv2d and the model surgery that installs it. Pre-registration §3, §4."""

import torch.nn as nn
import torch.nn.functional as F

from src.quantizer import ternarize


class TernaryConv2d(nn.Conv2d):
    """Conv2d whose weights are ternarized on every forward pass.

    The STE is vanilla identity: forward uses w_t, backward hands the gradient to
    self.weight unchanged. It is deliberately NOT masked or rescaled -- that would be a
    different estimator, which prereg §12 rules out. Saturation comes from clamping the
    latent weights to [-1, 1] after each optimizer step, which the training loop does.
    """

    def forward(self, x):
        w_t, degenerate = ternarize(self.weight)
        self._last_degenerate = degenerate.detach()  # read once per epoch by the tracker
        w_ste = self.weight + (w_t - self.weight).detach()
        return F.conv2d(x, w_ste, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def ternarize_model(model, expected_count=None):
    """Replace every Conv2d except the stem with TernaryConv2d, in place.

    The existing weight Parameter is carried over rather than reinitialized, so a ternary
    arm and a full-precision arm at the same seed start from bit-identical weights.

    Returns the list of swapped module names, in the order used for logging.
    """
    names = [
        name
        for name, m in model.named_modules()
        if isinstance(m, nn.Conv2d) and name != "conv1"  # stem stays fp32 (prereg §3)
    ]

    for name in names:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, attr)

        new = TernaryConv2d(
            old.in_channels, old.out_channels, old.kernel_size,
            stride=old.stride, padding=old.padding, dilation=old.dilation,
            groups=old.groups, bias=old.bias is not None,
        )
        new.weight = old.weight
        if old.bias is not None:
            new.bias = old.bias
        setattr(parent, attr, new)

    if expected_count is not None and len(names) != expected_count:
        raise RuntimeError(
            f"ternarized {len(names)} convs, expected {expected_count}. "
            "Architecture and the ternarization set have drifted apart."
        )
    return names
