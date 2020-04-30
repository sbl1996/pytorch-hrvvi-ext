import math

import torch
import torch.nn as nn

from horch.models.modules import get_activation, Conv2d
from horch.models.utils import profile


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, groups, base_width):
        super().__init__()

        D = math.floor(out_channels // self.expansion * (base_width / 64))

        self.conv1 = Conv2d(in_channels, D * groups, kernel_size=1,
                            norm_layer='default', activation='default')
        self.conv2 = Conv2d(D * groups, D * groups, kernel_size=3, stride=stride, groups=groups,
                            norm_layer='default', activation='default')
        self.conv3 = Conv2d(D * groups, out_channels, kernel_size=1,
                            norm_layer='default')
        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                               norm_layer='default') if stride != 1 or in_channels != out_channels else nn.Identity()
        self.relu = get_activation('default')

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        x = self.relu(x)
        return x


class ResNeXt(nn.Module):

    def __init__(self, stages=(64, 256, 512, 1024), depth=29, groups=8, base_width=64, num_classes=10):
        super().__init__()
        layers = [(depth - 2) // 9] * 3

        self.stages = stages

        self.conv = Conv2d(3, self.stages[0], kernel_size=3)

        self.layer1 = self._make_layer(
            self.stages[0], self.stages[1], layers[0], stride=1, groups=groups, base_width=base_width)
        self.layer2 = self._make_layer(
            self.stages[1], self.stages[2], layers[1], stride=2, groups=groups, base_width=base_width)
        self.layer3 = self._make_layer(
            self.stages[2], self.stages[3], layers[2], stride=2, groups=groups, base_width=base_width)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stages[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride, groups, base_width):
        layers = [Bottleneck(in_channels, out_channels, stride=stride, groups=groups, base_width=base_width)]
        for i in range(1, blocks):
            layers.append(
                Bottleneck(out_channels, out_channels, stride=1, groups=groups, base_width=base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_resnext():

    x = torch.randn(1, 3, 32, 32)

    # ResNeXt-29, 8×64d
    net = ResNeXt(stages=(64, 256, 512, 1024), depth=29, groups=8, base_width=64)
    assert profile(net, (x,))[1] == 34426634

    # ResNeXt-29, 16×64d
    net = ResNeXt(stages=(64, 256, 512, 1024), depth=29, groups=16, base_width=64)
    assert profile(net, (x,))[1] == 68155146
