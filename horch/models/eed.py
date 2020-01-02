import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.detection.enhance import BiFPN
from horch.models.modules import Conv2d, get_activation, Sequential


class WeightedFusion(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.full((n,), 1.0 / n), requires_grad=True)
        self.eps = 1e-4

    def forward(self, *xs):
        n = len(xs)
        assert n == self.weight.size(0)
        w = torch.relu(self.weight)
        w = w / (torch.sum(w, dim=0) + self.eps)
        x = 0
        for i in range(n):
            x += w[i] * xs[i]
        return x


class SideHead(nn.Module):

    def __init__(self, side_in_channels):
        super().__init__()
        n = len(side_in_channels)
        self.sides = nn.ModuleList([
            nn.Sequential(
                Conv2d(c, 1, 1, norm_layer='default')
            )
            for c in side_in_channels
        ])
        self.weight = nn.Parameter(torch.full((n,), 1.0 / n), requires_grad=True)
        self.eps = 1e-4

    def forward(self, *cs):
        size = cs[0].size()[2:4]
        w = torch.relu(self.weight)
        w = w / (torch.sum(w, dim=0) + self.eps)

        p = w[0] * self.sides[0](cs[0])
        for i in range(1, len(cs)):
            c = self.sides[i](cs[i])
            c = F.interpolate(c, size, mode='bilinear', align_corners=False)
            p += w[i] * c
        return p


class EED(nn.Module):
    def __init__(self, backbone, in_channels_list, f_channels=128, num_fpn_layers=2, deep_supervision=False, drop_rate=0.0):
        super().__init__()
        self.deep_supervison = deep_supervision
        self.backbone = backbone
        self.num_fpn_layers = num_fpn_layers
        n = len(in_channels_list)
        self.fpns = nn.ModuleList([
            BiFPN(in_channels_list, f_channels),
            *[BiFPN([f_channels] * n, f_channels)
              for _ in range(num_fpn_layers - 1)]
        ])
        self.heads = nn.ModuleList([
            SideHead([f_channels] * n)
            for _ in range(num_fpn_layers)
        ])
        self.weight = nn.Parameter(
            torch.full((self.num_fpn_layers,), 1.0 / self.num_fpn_layers), requires_grad=True)
        self.eps = 1e-4
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

        w = torch.relu(self.weight)
        w = w / (torch.sum(w, dim=0) + self.eps)

        outs = []
        fuse = 0
        for i, (fpn, head) in enumerate(zip(self.fpns, self.heads)):
            ps = fpn(*ps)
            p = head(*ps)
            outs.append(p)
            fuse += w[i] * p

        if self.training and self.deep_supervison:
            return outs, fuse
        else:
            return fuse
