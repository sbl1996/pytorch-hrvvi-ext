import torch
import torch.nn as nn

from horch.models.layers import Conv2d, DWConv2d
from horch.nn import HardSigmoid


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        channels = in_channels // reduction
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d(in_channels, channels, kernel_size=1, norm='bn', act='relu'),
            Conv2d(channels, in_channels, kernel_size=1, bias=False),
            HardSigmoid(True),
        )

    def forward(self, x):
        return x * self.layers(x)


def channel_shuffle(x):
    b, c, h, w = x.data.size()

    x = x.reshape(b * c // 2, 2, h * w)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, c // 2, h, w)
    return x[0], x[1]


class XceptionUnit(nn.Module):
    def __init__(self, in_channels, activation, use_se):
        super().__init__()
        channels = in_channels // 2
        branch = [
            DWConv2d(channels, channels, kernel_size=3,
                     norm='default', act=activation),
            DWConv2d(channels, channels, kernel_size=3,
                     norm='default', act=activation),
            DWConv2d(channels, channels, kernel_size=3,
                     norm='default', act=activation),
        ]
        if use_se:
            branch.append(SELayer(channels, reduction=2))
        self.branch = nn.Sequential(*branch)

    def forward(self, x):
        x_proj, x = channel_shuffle(x)
        return torch.cat((x_proj, self.branch(x)), dim=1)


class BasicUnit(nn.Module):
    def __init__(self, in_channels, kernel_size, activation, use_se):
        super().__init__()
        assert kernel_size in [3, 5, 7]
        channels = in_channels // 2
        branch = [
            Conv2d(channels, channels, kernel_size=1,
                   norm='default', act=activation),
            DWConv2d(channels, channels, kernel_size=kernel_size,
                     norm='default', act=activation),
        ]
        if use_se:
            branch.append(SELayer(channels, reduction=2))
        self.branch = nn.Sequential(*branch)

    def forward(self, x):
        x_proj, x = channel_shuffle(x)
        return torch.cat((x_proj, self.branch(x)), dim=1)


class ReduceUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, use_se):
        super().__init__()
        assert kernel_size in [3, 5, 7]
        mid_channels = out_channels // 2
        output = out_channels - in_channels
        branch_main = [
            Conv2d(in_channels, mid_channels, kernel_size=1,
                   norm='default', act=activation),
            DWConv2d(mid_channels, output, kernel_size=kernel_size, stride=2,
                     norm='default', act=activation),
        ]
        if use_se:
            branch_main.append(SELayer(output, reduction=2))
        self.branch_main = nn.Sequential(*branch_main)

        self.branch_proj = DWConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=2,
                                    norm='default', act=activation)

    def forward(self, x):
        return torch.cat((self.branch_proj(x), self.branch_main(x)), dim=1)


def _make_layer(block, num_units, in_channels, out_channels, stride, use_se):
    units = nn.Sequential()
    units.add_module("unit1",
                     ReduceUnit(in_channels, out_channels) if stride == 2 \
                         else Conv2d(in_channels, out_channels, kernel_size=3,
                                     norm='default', act='default'))
    for i in range(1, num_units):
        units.add_module(f"unit{i + 1}", block(out_channels, use_se))
    return units


# class ShuffleNetV2_Plus(nn.Module):
#     def __init__(self, stem_channels, channels_per_stage, units_per_stage, final_channels, num_classes=10, use_se=False):
#         super().__init__()
#         stage_out_channels = [16, 68, 168, 336, 672, 1280]
#         stage_repeats = [4, 4, 8, 4]
#         architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
#         self.stem = Conv2d(3, stem_channels, kernel_size=3,
#                            norm_layer='default', activation='default')
#
#         self.stage1 = _make_layer(block, units_per_stage[0], stem_channels, channels_per_stage[0], 1, use_se)
#         self.stage2 = _make_layer(block, units_per_stage[1], channels_per_stage[0], channels_per_stage[1], 2, use_se)
#         self.stage3 = _make_layer(block, units_per_stage[2], channels_per_stage[1], channels_per_stage[2], 2, use_se)
#         self.final_block = Conv2d(channels_per_stage[2], final_channels, kernel_size=1,
#                                   activation='default', norm_layer='default')
#         self.final_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(final_channels, num_classes)
#
#     def get_block(self, idx):
#
#
#     def forward(self, x):
#         x = self.stem(x)
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.final_block(x)
#         x = self.final_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


# def test_net():
#     net = ShuffleNetV2(32, [128, 256, 512], [4, 8, 4], 512, num_classes=10, use_se=True, residual=True)
#
#     x = torch.randn(2, 3, 32, 32)
#     y = net(x)
