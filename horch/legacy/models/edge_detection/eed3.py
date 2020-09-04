import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.bifpn import BiFPN, fast_normalize
from horch.models.modules import Conv2d


class SideHead(nn.Module):

    def __init__(self, side_in_channels, out_channels):
        super().__init__()
        n = len(side_in_channels)
        self.sides = nn.ModuleList([
            nn.Sequential(
                Conv2d(c, out_channels, 1, norm='default')
            )
            for c in side_in_channels
        ])
        self.weight = nn.Parameter(torch.ones((len(side_in_channels),)), requires_grad=True)

    def forward(self, *cs):
        size = cs[0].size()[2:4]

        w = fast_normalize(self.weight)

        p = w[0] * self.sides[0](cs[0])
        for i in range(1, len(cs)):
            c = self.sides[i](cs[i])
            c = F.interpolate(c, size, mode='bilinear', align_corners=False)
            p += w[i] * c
        return p


class SED(nn.Module):
    def __init__(self, backbone, in_channels_list, f_channels=128, num_fpn_layers=2,
                 drop_rate=0.0, out_channels=18):
        super().__init__()
        self.backbone = backbone
        self.num_fpn_layers = num_fpn_layers
        n = len(in_channels_list)
        self.num_levels = n
        self.lats = nn.ModuleList([
            Conv2d(c, f_channels, kernel_size=1, norm='bn')
            for c in in_channels_list
        ])
        self.fpns = nn.ModuleList([
            BiFPN([f_channels] * n, f_channels)
            for _ in range(num_fpn_layers)
        ])
        self.head = SideHead([f_channels] * n, out_channels)

        self.weights = nn.Parameter(
            torch.ones((self.num_fpn_layers + 1, self.num_levels)), requires_grad=True)
        self.dropout = nn.Dropout2d(drop_rate)

    def get_param_groups(self):
        group1 = self.backbone.parameters()
        layers = [
            self.fpn, self.head
        ]
        group2 = [
            p
            for l in layers
            for p in l.parameters()
        ]
        return [group1, group2]

    def forward(self, x):
        p1, p2, p3, _, p5 = self.backbone(x)
        ps = [p1, p2, p3, p5]

        ps = [
            self.dropout(p)
            for p in ps
        ]

        ps = [lat(p) for p, lat in zip(ps, self.lats)]

        ws = fast_normalize(self.weights, dim=1)

        fuses = [ws[0, i] * ps[i] for i in range(self.num_levels)]
        for i, fpn in enumerate(self.fpns):
            ps = fpn(*ps)
            for j in range(self.num_levels):
                fuses[j] += ws[i + 1, j] * ps[j]
        p = self.head(*fuses)
        return p
