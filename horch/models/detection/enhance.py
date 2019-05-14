import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import upsample_add, Conv2d


class TopDown(nn.Module):
    def __init__(self, in_channels, f_channels, norm_layer='gn'):
        super().__init__()
        self.lat = Conv2d(
            in_channels, f_channels, kernel_size=1,
            norm_layer=norm_layer)
        self.conv = Conv2d(
            f_channels, f_channels, kernel_size=3,
            norm_layer=norm_layer)

    def forward(self, c, p):
        p = upsample_add(p, self.lat(c))
        p = self.conv(p)
        return p


class FPN(nn.Module):
    r"""
    Feature Pyramid Network which enhance features of different levels.

    Parameters
    ----------
    in_channels : tuple of ints
        Number of input channels of every level, e.g., ``(256,512,1024)``
    out_channels : int
        Number of output channels.
    norm_layer : str
        `bn` for Batch Normalization and `gn` for Group Normalization.
        Default: `bn`
    """
    def __init__(self, in_channels, out_channels=256, norm_layer='gn'):
        super().__init__()
        self.lat = Conv2d(in_channels[-1], out_channels, kernel_size=1, norm_layer=norm_layer)
        self.topdowns = nn.ModuleList([
            TopDown(c, out_channels, norm_layer=norm_layer)
            for c in in_channels[:-1]
        ])

    def forward(self, *cs):
        ps = (self.lat(cs[-1]),)
        for c, topdown in zip(reversed(cs[:-1]), reversed(self.topdowns)):
            p = topdown(c, ps[0])
            ps = (p,) + ps
        return ps



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
