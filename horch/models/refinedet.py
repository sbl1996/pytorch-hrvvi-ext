import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.modules import Conv2d, get_activation


class TransferConnection(nn.Module):
    def __init__(self, in_channels, f_channels, last=False):
        super().__init__()
        kernel_size = 3
        self.last = last
        self.conv1 = nn.Sequential(
            Conv2d(in_channels, f_channels, kernel_size=kernel_size,
                   norm_layer='default', activation='default'),
            Conv2d(f_channels, f_channels, kernel_size=kernel_size,
                   norm_layer='default'),
        )
        # if not last:
        #     self.deconv = Conv2d(f_channels, f_channels, kernel_size=4, stride=2,
        #                          norm_layer='default', transposed=True)
        self.conv2 = nn.Sequential(
            get_activation('default'),
            Conv2d(f_channels, f_channels, kernel_size=kernel_size,
                   norm_layer='default', activation='default')
        )

    def forward(self, x, x_next=None):
        x = self.conv1(x)
        if not self.last:
            x_next = F.interpolate(x_next, x.size()[2:4], mode='bilinear', align_corners=False)
            x = x + x_next
        x = self.conv2(x)
        return x


class SideHead(nn.Module):
    def __init__(self, side_in_channels):
        super().__init__()
        self.sides = nn.ModuleList([
            Conv2d(c, 1, 1, norm_layer='default', activation='default')
            for c in side_in_channels
        ])
        self.pred = nn.Conv2d(len(side_in_channels), 1, 1)

    def forward(self, *cs):
        size = cs[0].size()[2:4]
        ps = [self.sides[0](cs[0])]
        for side, c in zip(self.sides[1:], cs[1:]):
            p = side(c)
            p = F.interpolate(p, size, mode='bilinear', align_corners=False)
            ps.append(p)
        p = torch.cat(ps, dim=1)
        p = self.pred(p)
        return p


class RefinEDet(nn.Module):
    def __init__(self, backbone, in_channels_list, f_channels=128):
        super().__init__()
        self.backbone = backbone
        self.r_head = SideHead(in_channels_list)
        self.tcbs = nn.ModuleList([
            TransferConnection(c, f_channels)
            for c in in_channels_list[:-1]
        ])
        self.tcbs.append(
            TransferConnection(in_channels_list[-1], f_channels, last=True)
        )
        self.d_head = SideHead([f_channels] * len(in_channels_list))
        self.pred = nn.Conv2d(2, 1, 1)

    def get_param_groups(self):
        group1 = self.backbone.parameters()
        layers = [
            self.side1, self.side2, self.side3, self.side5, self.pred
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
        r_pred = self.r_head(*cs)

        dcs = [self.tcbs[-1](cs[-1])]
        for c, tcb in zip(reversed(cs[:-1]), reversed(self.tcbs[:-1])):
            dcs.append(tcb(c, dcs[-1]))
        dcs.reverse()

        d_pred = self.d_head(*dcs)
        pred = torch.cat([r_pred, d_pred], dim=1)
        pred = self.pred(pred)
        return pred
