import torch
import torch.nn as nn

from horch.models.attention import SEModule
from horch.models.modules import Conv2d, get_activation


def channel_shuffle(x, groups=2):
    b, c, h, w = x.data.size()
    channels_per_group = c // groups

    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, -1, h, w)

    return x


class ResUnit(nn.Module):
    def __init__(self, in_channels, use_se=False):
        super().__init__()
        assert in_channels % 2 == 0
        channels = in_channels // 2
        self.branch = nn.Sequential(
            Conv2d(channels, channels, kernel_size=1,
                   activation='default', norm_layer='default'),
            Conv2d(channels, channels, kernel_size=3, groups=channels,
                   activation=None, norm_layer='default'),
            Conv2d(channels, channels, kernel_size=1,
                   activation=None, norm_layer='default'),
            SEModule(channels, reduction=2) if use_se else nn.Identity()
        )
        self.relu = get_activation()

    def forward(self, x):
        c = x.size(1) // 2
        x2 = x[:, c:, :, :]
        x2 = x2 + self.branch(x2)
        x2 = self.relu(x2)
        x = torch.cat([x[:, :c, :, :], x2], dim=1)
        x = channel_shuffle(x)
        return x


class BasicUnit(nn.Module):
    def __init__(self, in_channels, use_se=False):
        super().__init__()
        assert in_channels % 2 == 0
        channels = in_channels // 2
        self.branch = nn.Sequential(
            Conv2d(channels, channels, kernel_size=1,
                   activation='default', norm_layer='default'),
            Conv2d(channels, channels, kernel_size=3, groups=channels,
                   activation=None, norm_layer='default'),
            Conv2d(channels, channels, kernel_size=1,
                   activation='default', norm_layer='default'),
            SEModule(channels, reduction=2) if use_se else nn.Identity()
        )

    def forward(self, x):
        c = x.size(1) // 2
        x = torch.cat([x[:, :c, :, :], self.branch(x[:, c:, :, :])], dim=1)
        x = channel_shuffle(x)
        return x


class ReduceUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 2 == 0
        channels = out_channels // 2

        self.branch1 = nn.Sequential(
            Conv2d(in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels,
                   activation=None, norm_layer='default'),
            Conv2d(in_channels, channels, kernel_size=1,
                   activation='default', norm_layer='default'),
        )

        self.branch2 = nn.Sequential(
            Conv2d(in_channels, channels, kernel_size=1,
                   activation='default', norm_layer='default'),
            Conv2d(channels, channels, kernel_size=3, stride=2, groups=channels,
                   activation=None, norm_layer='default'),
            Conv2d(channels, channels, kernel_size=1,
                   activation='default', norm_layer='default'),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        x = channel_shuffle(x, 2)
        return x


def _make_layer(block, num_units, in_channels, out_channels, stride, use_se):
    units = nn.Sequential()
    units.add_module("unit1",
                     ReduceUnit(in_channels, out_channels) if stride == 2 \
                         else Conv2d(in_channels, out_channels, kernel_size=3,
                                     norm_layer='default', activation='default'))
    for i in range(1, num_units):
        units.add_module(f"unit{i + 1}", block(out_channels, use_se))
    return units


class ShuffleNetV2(nn.Module):
    def __init__(self, stem_channels, channels_per_stage, units_per_stage, final_channels, num_classes=10, use_se=False,
                 residual=False):
        super().__init__()
        self.stem = Conv2d(3, stem_channels, kernel_size=3,
                           activation='default', norm_layer='default')
        block = ResUnit if residual else BasicUnit
        self.stage1 = _make_layer(block, units_per_stage[0], stem_channels, channels_per_stage[0], 1, use_se)
        self.stage2 = _make_layer(block, units_per_stage[1], channels_per_stage[0], channels_per_stage[1], 2, use_se)
        self.stage3 = _make_layer(block, units_per_stage[2], channels_per_stage[1], channels_per_stage[2], 2, use_se)
        self.final_block = Conv2d(channels_per_stage[2], final_channels, kernel_size=1,
                                  activation='default', norm_layer='default')
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_block(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_net():
    net = ShuffleNetV2(32, [128, 256, 512], [4, 8, 4], 512, num_classes=10, use_se=True, residual=True)

    x = torch.randn(2, 3, 32, 32)
    y = net(x)