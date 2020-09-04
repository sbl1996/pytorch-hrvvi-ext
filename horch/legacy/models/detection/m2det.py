import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.common import tuplify
from horch.models.modules import Conv2d, upsample_add, upsample_concat, Identity
from horch.models.attention import SEModule


class TUM(nn.Module):
    r"""

    """

    def __init__(self, in_channels, f_channels=256, num_scales=6, no_padding=(-1, 0), lite=False):
        super().__init__()
        self.num_scales = num_scales
        self.down = nn.ModuleList([])
        self.lat = nn.ModuleList([
            Conv2d(in_channels, f_channels, kernel_size=3, stride=1,
                   norm='default', act='default', depthwise_separable=lite)
            if in_channels != f_channels else Identity()
        ])
        self.out = nn.ModuleList([
            Conv2d(f_channels, f_channels // 2, kernel_size=1, stride=1,
                   norm='default', act='default')
        ])
        for i in range(num_scales - 1):
            self.down.append(Conv2d(in_channels, f_channels, kernel_size=3, stride=2,
                                    norm='default', act='default', depthwise_separable=lite))
            self.lat.append(Conv2d(f_channels, f_channels, kernel_size=3, stride=1,
                                   norm='default', act='default', depthwise_separable=lite))
            self.out.append(Conv2d(f_channels, f_channels // 2, kernel_size=1, stride=1,
                                   norm='default', act='default'))
            in_channels = f_channels
        no_padding = tuplify(no_padding, 2)
        for i in range(no_padding[0], 0):
            l = self.down[i][0]
            if lite:
                l = l[0]
            p = l.padding
            l.padding = (0, p[1])
        for i in range(no_padding[1], 0):
            l = self.down[i][0]
            if lite:
                l = l[0]
            p = l.padding
            l.padding = (p[0], 0)

    def forward(self, x):
        xs = [x]
        for i in range(self.num_scales - 1):
            x = self.down[i](x)
            xs.append(x)
        xs[0] = self.lat[0](xs[0])

        outs = []
        x = xs[-1]
        outs.append(self.out[-1](x))
        for i in reversed(range(self.num_scales - 1)):
            x = upsample_add(self.lat[i+1](x), xs[i])
            outs.append(self.out[i](x))
        outs.reverse()
        return outs


class FFMv1(nn.Module):
    def __init__(self, in_channels_list, f_channels, lite=False):
        super().__init__()
        self.conv1 = Conv2d(in_channels_list[0], f_channels, kernel_size=1,
                            norm='default', act='default')
        self.conv2 = Conv2d(in_channels_list[1], f_channels * 2, kernel_size=3,
                            norm='default', act='default', depthwise_separable=lite)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        return upsample_concat(x2, x1)


class FFMv2(nn.Module):
    def __init__(self, in_channels, f_channels):
        super().__init__()
        self.conv = Conv2d(in_channels, f_channels, kernel_size=1,
                           norm='default', act='default')

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        return torch.cat((x1, x2), dim=1)


class SFAM(nn.Module):
    def __init__(self, in_channels_list, reduction=16):
        super().__init__()
        self.se = nn.ModuleList([
            SEModule(c, reduction)
            for c in in_channels_list
        ])

    def forward(self, *css):
        ps = []
        for cs, se in zip(zip(*css), self.se):
            p = torch.cat(cs, dim=1)
            p = se(p)
            ps.append(p)
        return ps


class M2Det(nn.Module):
    r"""Implements M2Det (https://arxiv.org/pdf/1811.04533.pdf)

    Parameters
    ----------
    in_channels_list : sequence of ints
        Number of input channels of every level, e.g., ``(60, 120)``
        Notice: Inputs must be c3, c4.
    num_scales : int
        Number of scales.
        For example, if num_scales is 4, outputs will be [p3, p4, p5, p6].
    num_levels : int
        Number of stacked TUMs.
        More levels give better performance but lead to more parameters.
    f_channels : int
        Number of feature channels.
    no_padding : int
        Do not pad from which layers.
        For example, if num_scales is 4 and no_padding is -2,
        p5 and p6 is produced by a conv3x3-s2x2 without padding.
    se_reduction : int
        Reduction of SEModule.
    lite : bool
        Whether to replace conv3x3 with depth-wise seprable conv.
    """

    def __init__(self, in_channels_list, num_scales, num_levels,
                 f_channels=256, no_padding=0, se_reduction=16, lite=False):
        super().__init__()
        assert len(in_channels_list) == 2
        assert num_levels >= 2
        self.ffm1 = FFMv1(in_channels_list, f_channels, lite=lite)
        self.tums = nn.ModuleList([
            TUM(f_channels * 3, f_channels, num_scales, no_padding, lite)
        ])
        for _ in range(num_levels - 1):
            self.tums.append(
                TUM(f_channels, f_channels, num_scales, no_padding, lite)
            )
        self.ffm2 = nn.ModuleList([
            FFMv2(f_channels * 3, f_channels // 2)
            for _ in range(num_levels - 1)
        ])
        self.se = nn.ModuleList([
            SEModule(f_channels // 2 * num_levels, se_reduction)
            for _ in range(num_scales)
        ])
        self.out_channels = [ f_channels // 2 * num_levels ] * num_scales

    def forward(self, c3, c4):
        c = self.ffm1(c3, c4)
        css = [self.tums[0](c)]
        for ffm, tum in zip(self.ffm2, self.tums[1:]):
            p = ffm(c, css[-1][0])
            css.append(tum(p))
        ps = []
        for cs, se in zip(zip(*css), self.se):
            p = torch.cat(cs, dim=1)
            p = se(p)
            ps.append(p)
        return ps
