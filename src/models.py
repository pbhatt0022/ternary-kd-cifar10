"""CIFAR-adapted ResNets. Pre-registration §3 and §6.

Two families:
  * ResNet-18 / ResNet-34 (student / teacher): torchvision BasicBlock topology with the
    7x7 stride-2 stem replaced by 3x3 stride-1 and the initial maxpool removed. Projection
    (option B) shortcuts, i.e. the 1x1 downsample convs prereg §3 names as ternarizable.
  * The He 6n+2 family for arm E, with option A parameter-free shortcuts, as in He et al.
    (2016) §4.2 for CIFAR.
"""

import torch.nn as nn
import torch.nn.functional as F

# ResNet-18 has 19 convs outside the stem: 16 in blocks + 3 downsample convs (prereg §3).
EXPECTED_TERNARY_CONVS = {"resnet18_cifar": 19, "resnet34_cifar": 39}


class ShortcutA(nn.Module):
    """He et al. option A: stride subsample, zero-pad the new channels. No parameters."""

    def __init__(self, stride, extra_channels):
        super().__init__()
        self.stride = stride
        self.pad_lo = extra_channels // 2
        self.pad_hi = extra_channels - self.pad_lo

    def forward(self, x):
        x = x[:, :, :: self.stride, :: self.stride]
        return F.pad(x, (0, 0, 0, 0, self.pad_lo, self.pad_hi))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option_a=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            if option_a:
                self.downsample = ShortcutA(stride, planes - in_planes)
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


def init_weights(model):
    """prereg §3. Note what is deliberately absent: there is NO zero-gamma init of the
    final BN in each residual block (prereg §12). Every BN starts at gamma=1, beta=0.
    FC keeps the PyTorch default."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class ResNetCifar(nn.Module):
    """Stage i>0 downsamples at its first block; stage 0 keeps 32x32."""

    def __init__(self, blocks_per_stage, channels, option_a=False, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels[0], 3, 1, 1, bias=False)  # 3x3 s1, no maxpool
        self.bn1 = nn.BatchNorm2d(channels[0])

        in_planes = channels[0]
        self.n_stages = len(blocks_per_stage)
        for i, (n_blocks, ch) in enumerate(zip(blocks_per_stage, channels)):
            blocks = []
            for b in range(n_blocks):
                stride = 2 if (i > 0 and b == 0) else 1
                blocks.append(BasicBlock(in_planes, ch, stride, option_a))
                in_planes = ch
            setattr(self, f"layer{i + 1}", nn.Sequential(*blocks))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)
        init_weights(self)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for i in range(1, self.n_stages + 1):
            x = getattr(self, f"layer{i}")(x)
        return self.fc(self.avgpool(x).flatten(1))


def resnet18_cifar(num_classes=10):
    return ResNetCifar([2, 2, 2, 2], [64, 128, 256, 512], num_classes=num_classes)


def resnet34_cifar(num_classes=10):
    return ResNetCifar([3, 4, 6, 3], [64, 128, 256, 512], num_classes=num_classes)


def resnet6n2(n, num_classes=10):
    """n in {3,5,7,9,18} -> ResNet-20/32/44/56/110. Arm E candidates (prereg §6)."""
    return ResNetCifar([2 * n] * 3, [16, 32, 64], option_a=True, num_classes=num_classes)


ARM_E_CANDIDATES = {20: 3, 32: 5, 44: 7, 56: 9, 110: 18}  # depth -> n
