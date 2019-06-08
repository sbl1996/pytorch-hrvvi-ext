import torch
import torch.nn as nn

from horch.models.modules import Conv2d, Pool, Identity

class PairwiseCombination(nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, x0, x1):
        return self.left(x0) + self.right(x1)


class NormalCell(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super().__init__()
        self.p2 = PairwiseCombination(
            Pool('avg', 3),
            Pool('max', 3),
        )
        self.p3 = PairwiseCombination(
            Identity(),
            Pool('avg', 3),
        )
        self.p4 = PairwiseCombination(
            Conv2d(in_channels1, out_channels, 5, depthwise_separable=True),
            Conv2d(in_channels2, out_channels, 3, depthwise_separable=True),
        )
        self.p5 = PairwiseCombination(
            Conv2d(in_channels1, out_channels, 3, depthwise_separable=True),
            Conv2d(in_channels2, out_channels, 3, depthwise_separable=True),
        )

    def forward(self, x0, x1):
        return x0
