import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.detection.enhance import FPN, FPN2
from horch.models.modules import Conv2d, get_activation


class SideHead(nn.Module):
    def __init__(self, side_in_channels):
        super().__init__()
        self.sides = nn.ModuleList([
            Conv2d(c, 1, 1, norm_layer='default', activation='default')
            for c in side_in_channels
        ])
        self.fuse = nn.Conv2d(len(side_in_channels), 1, 1, bias=False)
        nn.init.constant_(self.fuse.weight, 1 / (len(side_in_channels)))

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


class PANet(nn.Module):
    def __init__(self, backbone, in_channels_list, f_channels=128):
        super().__init__()
        self.backbone = backbone
        self.fpn1 = FPN(in_channels_list, f_channels, upsample='deconv', aggregate='cat')
        self.fpn2 = FPN2([f_channels] * len(in_channels_list), f_channels)

        self.head = SideHead([f_channels] * len(in_channels_list))

    def forward(self, x):
        c1, c2, c3, _, c5 = self.backbone(x)
        cs = [c1, c2, c3, c5]

        ps = self.fpn1(*cs)
        ps = self.fpn2(*ps)

        ps, p = self.head(*ps)
        return p
