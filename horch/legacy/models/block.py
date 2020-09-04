from horch.models.modules import Norm, Conv2d
from horch.models.attention import SEModule
from horch.models.cifar.pyramidnet import Bottleneck as PyrUnit
from torch import nn as nn

from pytorchcv.models.shufflenetv2b import ShuffleUnit


def shuffle_block(in_channels, out_channels):
    return ShuffleUnit(in_channels, out_channels, downsample=True, use_se=True, use_residual=False, shuffle_group_first=True)


def pyramid_block(in_channels, out_channels):
    assert in_channels <= out_channels
    return PyrUnit(in_channels, out_channels // PyrUnit.expansion, stride=2)


def mb_conv_block(in_channels, out_channels, expand_ratio=4, kernel_size=3):
    if expand_ratio == 1:
        return MBConv(in_channels, in_channels, out_channels, kernel_size=kernel_size, stride=2)
    else:
        return MBConv(in_channels, out_channels * expand_ratio, out_channels, kernel_size=kernel_size, stride=2)


class MBConv(nn.Sequential):
    def __init__(self, in_channels, channels, out_channels, kernel_size, stride=1, se_ratio=1 / 16):
        super().__init__()

        self.bn = Norm('default', in_channels)
        if in_channels != channels:
            self.expand = Conv2d(in_channels, channels, kernel_size=1,
                                 norm='default', act='default')

        self.dwconv = Conv2d(channels, channels, kernel_size, stride=stride, groups=channels,
                             norm='default', act='default')

        if se_ratio:
            assert 0 < se_ratio < 1
            self.se = SEModule(channels, reduction=int(1 / se_ratio))

        if out_channels is not None:
            self.project = Conv2d(channels, out_channels, kernel_size=1,
                                  norm='default')
        self.use_res_connect = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        x = super().forward(x)
        if self.use_res_connect:
            x += identity
        return x