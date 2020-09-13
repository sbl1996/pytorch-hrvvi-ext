import torch
from torch import nn as nn
from torch.nn import functional as F

from horch.models.layers import DWConv2d
from horch.models.modules import Conv2d


def fast_normalize(w, eps=1e-4, dim=0):
    w = torch.relu(w)
    w = w / (torch.sum(w, dim=dim, keepdim=True) + eps)
    return w


class BottomUpFusion2(nn.Module):
    def __init__(self, f_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((2,)), requires_grad=True)
        self.conv = DWConv2d(f_channels, f_channels, kernel_size=3,
                           norm='def', act='def')

    def forward(self, p, pp):
        pp = F.max_pool2d(pp, kernel_size=2, ceil_mode=True)
        w = fast_normalize(self.weight)
        p = w[0] * p + w[1] * pp
        p = self.conv(p)
        return p


class TopDownFusion2(nn.Module):
    def __init__(self, f_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((2,)), requires_grad=True)
        self.conv = DWConv2d(f_channels, f_channels, kernel_size=3,
                           norm='def', act='def')

    def forward(self, p, pp):
        h, w = p.size()[2:4]
        pp = F.interpolate(pp, (h, w), mode='bilinear', align_corners=False)
        w = fast_normalize(self.weight)
        p = w[0] * p + w[1] * pp
        p = self.conv(p)
        return p


def use_ceil_mode(xs, xl):
    ws, hs = xs.size()[2:4]
    wl, hl = xl.size()[2:4]
    return (wl / ws < 2) or (hl / hs < 2)


class BottomUpFusion3(nn.Module):
    def __init__(self, f_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((3,)), requires_grad=True)
        self.conv = DWConv2d(f_channels, f_channels, kernel_size=3,
                           norm='def', act='def')

    def forward(self, p1, p2, pp):
        pp = F.max_pool2d(pp, kernel_size=2, ceil_mode=use_ceil_mode(p1, pp))
        w = fast_normalize(self.weight)
        p = w[0] * p1 + w[1] * p2 + w[2] * pp
        p = self.conv(p)
        return p


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, f_channels):
        super().__init__()
        n = len(in_channels_list)
        self.lats = nn.ModuleList([
            Conv2d(c, f_channels, kernel_size=1, norm='def')
            if c != f_channels else nn.Identity()
            for c in in_channels_list
        ])
        self.tds = nn.ModuleList([
            TopDownFusion2(f_channels)
            for _ in range(n - 1)
        ])
        self.bus = nn.ModuleList([
            BottomUpFusion3(f_channels)
            for _ in range(n - 2)
        ])
        self.bu = BottomUpFusion2(f_channels)

    def forward(self, *ps):
        ps = [lat(p) for p, lat in zip(ps, self.lats)]

        ps2 = [ps[-1]]
        for p, td in zip(reversed(ps[:-1]), self.tds):
            ps2.append(td(p, ps2[-1]))
        ps3 = [ps2[-1]]
        ps2 = reversed(ps2[1:-1])

        for p1, p2, bu in zip(ps[1:-1], ps2, self.bus):
            ps3.append(bu(p1, p2, ps3[-1]))
        ps3.append(self.bu(ps[-1], ps3[-1]))
        return tuple(ps3)