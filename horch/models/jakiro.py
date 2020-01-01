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
            Conv2d(c, 1, 1, norm_layer='default')
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
        return p


class JakiroNet(nn.Module):
    def __init__(self, backbone, in_channels_list, f_channels=128, deep_supervision=False, drop_rate=0.0):
        super().__init__()
        self.backbone = backbone
        self.deep_supervision = deep_supervision
        self.tcbs = nn.ModuleList([
            TransferConnection(c, f_channels)
            for c in in_channels_list[:-1]
        ])
        self.tcbs.append(
            TransferConnection(in_channels_list[-1], f_channels, last=True)
        )
        self.classifier = Conv2d(f_channels, 2, 1)
        self.head1 = SideHead([f_channels] * len(in_channels_list))
        self.head2 = SideHead([f_channels] * len(in_channels_list))
        self.dropout = nn.Dropout2d(drop_rate)

    def get_param_groups(self):
        group1 = self.backbone.parameters()
        layers = [
            *self.tcbs, self.head
        ]
        group2 = [
            p
            for l in layers
            for p in l.parameters()
        ]
        return [group1, group2]

    def forward(self, x):
        b = len(x)
        c1, c2, c3, _, c5 = self.backbone(x)
        cs = [c1, c2, c3, c5]

        cs = [
            self.dropout(c)
            for c in cs
        ]

        dcs = [self.tcbs[-1](cs[-1])]
        for c, tcb in zip(reversed(cs[:-1]), reversed(self.tcbs[:-1])):
            dcs.append(tcb(c, dcs[-1]))
        dcs.reverse()

        dcs = [
            self.dropout(c)
            for c in dcs
        ]

        pc = self.classifier(dcs[-1])
        pc = F.adaptive_avg_pool2d(pc, 1).view(b, 2)
        c = pc.argmax(dim=1)
        ps = []
        for i in range(b):
            i_dcs = [c[[i]] for c in dcs]
            if c[i] == 0:
                p = self.head1(*i_dcs)
            else:
                p = self.head2(*i_dcs)
            ps.append(p)
        p = torch.cat(ps, dim=0)
        return p
