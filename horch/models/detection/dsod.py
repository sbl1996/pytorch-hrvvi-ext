import torch
import torch.nn as nn
import torch.nn.functional as F
from horch.models.modules import Conv2d


class DenseSupervision(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.left = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=1,
                   norm_layer='default', activation='default'),
            Conv2d(out_channels, out_channels, kernel_size=3, stride=2,
                   norm_layer='default', activation='default')
        )

        self.right = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv2d(in_channels, out_channels, kernel_size=1,
                   norm_layer='default', activation='default'),
        )

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return torch.cat([left, right], 1)


class DSOD(nn.Module):
    def __init__(self, in_channels_list, f_channels=256, num_extra_layers=4, no_padding=-2):
        super().__init__()
        self.trans = Conv2d(in_channels_list[-1], f_channels, kernel_size=1,
                            norm_layer='default', activation='default')
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv2d(in_channels_list[-2], f_channels, kernel_size=1,
                   norm_layer='default', activation='default')
        )
        in_channels_list = list(in_channels_list)
        in_channels_list[-1] = f_channels * 2
        self.dss = nn.ModuleList([
            DenseSupervision(in_channels_list[-1], f_channels)
        ])
        in_channels_list.append(f_channels * 2)
        for i in range(num_extra_layers - 1):
            self.dss.append(DenseSupervision(in_channels_list[-1], f_channels // 2))
            in_channels_list.append(f_channels)

        self.out_channels = in_channels_list
        for i in range(no_padding, 0):
            ds = self.dss[i]
            conv = ds.left[-1][0]
            conv.stride = (1, 1)
            conv.padding = (0, 0)
            pool = ds.right[0]
            pool.kernel_size = (3, 3)
            pool.stride = (1, 1)

    def forward(self, c3, c4):
        ps = [c3]
        p4 = torch.cat([self.trans(c4), self.down(c3)], dim=1)
        ps.append(p4)
        for ds in self.dss:
            ps.append(ds(ps[-1]))
        return ps
