import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.detection.enhance import BiFPN
from horch.models.modules import Conv2d, get_activation, Sequential


class SideHead(nn.Module):

    def __init__(self, side_in_channels):
        super().__init__()
        self.sides = nn.ModuleList([
            nn.Sequential(
                Conv2d(c, 1, 1, norm_layer='default')
            )
            for c in side_in_channels
        ])
        self.fuse = nn.Conv2d(len(side_in_channels), 1, 1)
        nn.init.constant_(self.fuse.weight, 1 / (len(side_in_channels)))
        nn.init.constant_(self.fuse.bias, 0)

    def forward(self, *cs):
        size = cs[0].size()[2:4]
        ps = [self.sides[0](cs[0])]
        for side, c in zip(self.sides[1:], cs[1:]):
            p = side(c)
            p = F.interpolate(p, size, mode='bilinear', align_corners=False)
            ps.append(p)

        p = torch.cat(ps, dim=1)
        p = self.fuse(p)
        return ps, p


class ConvSideHead(nn.Module):

    def __init__(self, side_in_channels, num_layers=2):
        super().__init__()
        f_channels = side_in_channels[0]
        self.conv = nn.Sequential(*[
            Conv2d(f_channels, f_channels, kernel_size=3,
                   norm_layer='default', activation='relu')
            for _ in range(num_layers)
        ])

        self.sides = nn.ModuleList([
            Conv2d(f_channels, 1, 1, norm_layer='default')
            for c in side_in_channels
        ])
        self.fuse = nn.Conv2d(len(side_in_channels), 1, 1)
        nn.init.constant_(self.fuse.weight, 1 / (len(side_in_channels)))
        nn.init.constant_(self.fuse.bias, 0)

    def forward(self, *cs):
        size = cs[0].size()[2:4]
        cs = [
            self.conv(c)
            for c in cs
        ]

        ps = [self.sides[0](cs[0])]
        for side, c in zip(self.sides[1:], cs[1:]):
            p = side(c)
            p = F.interpolate(p, size, mode='bilinear', align_corners=False)
            ps.append(p)

        p = torch.cat(ps, dim=1)
        p = self.fuse(p)
        return ps, p


class EED(nn.Module):
    def __init__(self, backbone, in_channels_list, f_channels=128, num_fpn_layers=2, drop_rate=0.0):
        super().__init__()
        self.backbone = backbone
        n = len(in_channels_list)
        self.fpn = Sequential(
            BiFPN(in_channels_list, f_channels),
            *[BiFPN([f_channels] * n, f_channels)
              for _ in range(num_fpn_layers - 1)]
        )
        self.head = SideHead([f_channels] * n)
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
        c1, c2, c3, _, c5 = self.backbone(x)
        cs = [c1, c2, c3, c5]

        cs = [
            self.dropout(c)
            for c in cs
        ]

        ps = self.fpn(*cs)

        ps, p = self.head(*ps)
        return p
