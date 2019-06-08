import torch
import torch.nn as nn

from horch.models.modules import Conv2d, Pool, get_activation, Identity as IdentityM


class PairwiseCombination(nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, x0, x1):
        return self.left(x0) + self.right(x1)


def DWConv2d(in_channels, out_channels, kernel_size, stride=1):
    return nn.Sequential(
        get_activation('default'),
        *Conv2d(in_channels, out_channels, kernel_size, stride, depthwise_separable=True),
        nn.BatchNorm2d(out_channels),
    )


class DoubleAvgPool(nn.Module):
    def __init__(self, stride=1):
        super().__init__()
        self.pool = Pool('avg', 3, stride=stride)

    def forward(self, x, x1):
        x = self.pool(x)
        return x + x


class Identity(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super().__init__()
        self.conv = IdentityM()
        if in_channels is not None and in_channels != out_channels:
            self.conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        return self.conv(x)


class NormalCell(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        prev_channels = channels // 2 if stride == 2 else channels
        self.p1 = PairwiseCombination(
            DWConv2d(channels, channels, 3),
            Identity(),
        )
        self.p2 = PairwiseCombination(
            DWConv2d(prev_channels, channels, 3, stride=stride),
            DWConv2d(channels, channels, 5),
        )
        self.p3 = PairwiseCombination(
            Pool('avg', 3),
            Identity(prev_channels, channels),
        )
        self.p4 = DoubleAvgPool(stride=stride)
        self.p5 = PairwiseCombination(
            DWConv2d(prev_channels, channels, 5, stride=stride),
            DWConv2d(prev_channels, channels, 3, stride=stride),
        )

    def forward(self, h_prev, h):
        h1 = self.p1(h, h)
        h2 = self.p2(h_prev, h)
        h3 = self.p3(h, h_prev)
        h4 = self.p4(h_prev, h_prev)
        h5 = self.p5(h_prev, h_prev)
        return torch.cat([h1, h2, h3, h4, h5], dim=1)
