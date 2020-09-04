import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.modules import Conv2d


class DenseSupervision(nn.Module):

    def __init__(self, in_channels, f_channels):
        super().__init__()
        self.left = nn.Sequential(
            Conv2d(in_channels, f_channels, kernel_size=1,
                   norm='default', act='default'),
            Conv2d(f_channels, f_channels, kernel_size=3, stride=2,
                   norm='default', act='default')
        )

        self.right = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv2d(in_channels, f_channels, kernel_size=1,
                   norm='default', act='default'),
        )

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return torch.cat([left, right], 1)


class Transition(nn.Module):
    def __init__(self, in_channels1, in_channels2, f_channels):
        super().__init__()
        self.left = Conv2d(in_channels1, f_channels, kernel_size=1,
                           norm='default', act='default')
        self.right = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv2d(in_channels2, f_channels, kernel_size=1,
                   norm='default', act='default'),
        )

    def forward(self, x2, x1):
        x = torch.cat([self.left(x2), self.right(x1)], dim=1)
        return x


class DSOD(nn.Module):
    def __init__(self, in_channels_list, f_channels=256, num_extra_layers=4, no_padding=-2):
        super().__init__()
        out_channels = [in_channels_list[0]]
        self.trans = nn.ModuleList([
            Transition(in_channels_list[1], in_channels_list[0], f_channels)
        ])
        out_channels.append(f_channels * 2)
        for c in in_channels_list[2:]:
            self.trans.append(Transition(c, f_channels * 2, f_channels))
            out_channels.append(f_channels * 2)
        self.dss = nn.ModuleList([
            DenseSupervision(out_channels[-1], f_channels)
        ])
        out_channels.append(f_channels * 2)
        for i in range(num_extra_layers - 1):
            self.dss.append(DenseSupervision(out_channels[-1], f_channels // 2))
            out_channels.append(f_channels)

        self.out_channels = out_channels
        for i in range(no_padding, 0):
            ds = self.dss[i]
            conv = ds.left[-1][0]
            conv.stride = (1, 1)
            conv.padding = (0, 0)
            pool = ds.right[0]
            pool.kernel_size = (3, 3)
            pool.stride = (1, 1)

    def forward(self, *cs):
        ps = [cs[0]]
        for i in range(len(cs) - 1):
            ps.append(self.trans[i](cs[i+1], ps[-1]))
        for ds in self.dss:
            ps.append(ds(ps[-1]))
        return ps
