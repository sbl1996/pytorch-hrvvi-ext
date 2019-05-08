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
