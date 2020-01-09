import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import Conv2d


class Tri2Node(nn.Module):
    def __init__(self, f_channels):
        super().__init__()
        self.deconv = Conv2d(f_channels, f_channels, 2, 2, transposed=True,
                           norm_layer='default', activation='relu')
        self.conv = Conv2d(f_channels, f_channels, kernel_size=3,
                           norm_layer='default', activation='default')
        self.weight = nn.Parameter(torch.ones(2), requires_grad=True)

    def forward(self, pl, plt):
        w = torch.relu(self.weight)
        w = w / (torch.sum(w, dim=0) + 1e-4)

        p = w[0] * pl + w[1] * self.deconv(plt)
        p = self.conv(p)
        return p


class Tri3Node(nn.Module):
    def __init__(self, f_channels):
        super().__init__()
        self.deconv_lt = Conv2d(f_channels, f_channels, 2, 2, transposed=True,
                           norm_layer='default', activation='relu')

        self.deconv_t = Conv2d(f_channels, f_channels, 2, 2, transposed=True,
                           norm_layer='default', activation='relu')

        self.conv = Conv2d(f_channels, f_channels, kernel_size=3,
                           norm_layer='default', activation='default')
        self.weight = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, pl, plt, pt):
        w = torch.relu(self.weight)
        w = w / (torch.sum(w, dim=0) + 1e-4)
        p = w[0] * pl + w[1] * self.deconv_lt(plt) + w[2] * self.deconv_t(pt)
        p = self.conv(p)
        return p


class UBlock(nn.Module):
    def __init__(self, f_channels, num_levels=4):
        super().__init__()
        self.f_channels = f_channels
        self.num_levels = num_levels
        self.nodes = nn.ModuleList([
            nn.ModuleList([
                Tri2Node(f_channels),
                *[Tri3Node(f_channels) for _ in range(l - 1)],
                Tri2Node(f_channels),
            ])
            for l in range(1, num_levels)
        ])

    def forward(self, *ps):
        s = [ps[-1]]
        for l in range(1, self.num_levels):
            nodes = self.nodes[l-1]
            ns = [nodes[0](ps[-l-1], s[0])]
            for j in range(1, l):
                ns.append(nodes[j](ns[-1], s[j-1], s[j]))
            ns.append(nodes[-1](ns[-1], s[-1]))
            s = ns
        return s[-1]


class UPP(nn.Module):

    def __init__(self, backbone, in_channels_list, f_channels=128, drop_rate=0.0):
        super().__init__()
        self.backbone = backbone
        n = len(in_channels_list)
        self.num_levels = n
        self.lats = nn.ModuleList([
            Conv2d(c, f_channels, kernel_size=1, norm_layer='default')
            for c in in_channels_list
        ])
        self.ublock = UBlock(f_channels, len(in_channels_list))
        self.head = Conv2d(f_channels, 1, kernel_size=1)

        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        p1, p2, p3, _, p5 = self.backbone(x)
        ps = [p1, p2, p3, p5]

        ps = [
            self.dropout(p)
            for p in ps
        ]

        ps = [lat(p) for p, lat in zip(ps, self.lats)]
        p = self.ublock(*ps)
        p = self.head(p)
        return p
