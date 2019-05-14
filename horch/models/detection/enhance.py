import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import upsample_add, Conv2d, Sequential


class TopDown(nn.Module):
    def __init__(self, in_channels, f_channels, norm_layer='gn', lite=False):
        super().__init__()
        self.lat = Conv2d(
            in_channels, f_channels, kernel_size=1,
            norm_layer=norm_layer)
        self.conv = Conv2d(
            f_channels, f_channels, kernel_size=3,
            norm_layer=norm_layer, activation='default', depthwise_separable=lite)

    def forward(self, c, p):
        p = upsample_add(p, self.lat(c))
        p = self.conv(p)
        return p


class FPN(nn.Module):
    r"""
    Feature Pyramid Network which enhance features of different levels.

    Parameters
    ----------
    in_channels : sequence of ints
        Number of input channels of every level, e.g., ``(256,512,1024)``
    out_channels : int
        Number of output channels.
    norm_layer : str
        `bn` for Batch Normalization and `gn` for Group Normalization.
        Default: `bn`
    """
    def __init__(self, in_channels, out_channels=256, norm_layer='gn', lite=False):
        super().__init__()
        self.lat = Conv2d(in_channels[-1], out_channels, kernel_size=1, norm_layer=norm_layer)
        self.topdowns = nn.ModuleList([
            TopDown(c, out_channels, norm_layer=norm_layer, lite=lite)
            for c in in_channels[:-1]
        ])

    def forward(self, *cs):
        ps = (self.lat(cs[-1]),)
        for c, topdown in zip(reversed(cs[:-1]), reversed(self.topdowns)):
            p = topdown(c, ps[0])
            ps = (p,) + ps
        return ps


class BottomUp(nn.Module):
    def __init__(self, f_channels, norm_layer='gn', lite=False):
        super().__init__()
        self.down = Conv2d(
            f_channels, f_channels, kernel_size=3, stride=2,
            norm_layer=norm_layer, activation='default', depthwise_separable=lite,
        )
        self.conv = Conv2d(
            f_channels, f_channels, kernel_size=3,
            norm_layer=norm_layer, activation='default', depthwise_separable=lite)

    def forward(self, p, n):
        n = p + self.down(n)
        n = self.conv(n)
        return n


class FPN2(nn.Module):
    r"""
    Bottom-up path augmentation.

    Parameters
    ----------
    in_channels : sequence of ints
        Number of input channels of every level, e.g., ``(256,256,256)``
        Notice: they must be the same.
    out_channels : int
        Number of output channels.
    norm_layer : str
        `bn` for Batch Normalization and `gn` for Group Normalization.
        Default: `bn`
    """
    def __init__(self, in_channels, out_channels, norm_layer='gn', lite=False):
        super().__init__()
        assert len(set(in_channels)) == 1, "Input channels of every level must be the same"
        assert in_channels[0] == out_channels, "Input channels must be the same as `out_channels`"
        self.bottomups = nn.ModuleList([
            BottomUp(out_channels, norm_layer=norm_layer, lite=lite)
            for _ in in_channels[1:]
        ])

    def forward(self, *ps):
        ns = [ps[0]]
        for p, bottomup in zip(ps[1:], self.bottomups):
            n = bottomup(p, ns[-1])
            ns.append(n)
        return tuple(ns)


class ContextEnhance(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer='bn'):
        super().__init__()
        self.lats = nn.ModuleList([
            Conv2d(c, out_channels, kernel_size=1, norm_layer=norm_layer)
            for c in in_channels
        ])
        self.lat_glb = Conv2d(in_channels[-1], out_channels, kernel_size=1,
                              norm_layer=norm_layer)

    def forward(self, *cs):
        print(len(cs))
        size = cs[0].size()[2:4]
        p = self.lats[0](cs[0])
        for c, lat in zip(cs[1:], self.lats[1:]):
            p += F.interpolate(lat(c), size=size, mode='bilinear', align_corners=False)
        c_glb = F.adaptive_avg_pool2d(cs[-1], 1)
        p_glb = self.lat_glb(c_glb)
        p += p_glb
        return p


def stacked_fpn(num_stacked, in_channels, f_channels=256, norm_layer='bn', lite=False):
    r"""
    Stacked FPN with alternant top down block and bottom up block.

    Parameters
    ----------
    num_stacked : int
        Number of stacked fpns.
    in_channels : sequence of ints
        Number of input channels of every level, e.g., ``(128,256,512)``
    f_channels : int
        Number of feature (output) channels.
        Default: 256
    norm_layer : str
        `bn` for Batch Normalization and `gn` for Group Normalization.
        Default: "bn"
    lite : bool
        Whether to replace conv3x3 with depthwise seperable conv.
        Default: False
    """
    assert num_stacked >= 2, "Use FPN directly if `num_stacked` is smaller than 2."
    num_levels = len(in_channels)
    layers = [FPN(in_channels, f_channels, norm_layer=norm_layer, lite=lite)]
    for i in range(1, num_stacked):
        if i % 2 == 0:
            layers.append(FPN([f_channels] * num_levels, f_channels, norm_layer=norm_layer, lite=lite))
        else:
            layers.append(FPN2([f_channels] * num_levels, f_channels, norm_layer=norm_layer, lite=lite))
    return Sequential(*layers)