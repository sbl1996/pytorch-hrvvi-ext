from functools import partial

import torch.nn as nn

from horch.nn import DropPath
from horch.models.layers import Seq

from horch.nas.operations import OPS
from horch.nas.primitives import get_primitives
from horch.nas.darts.search import darts


class MixedOp(nn.Module):

    def __init__(self, C, stride, drop_path=0):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in get_primitives():
            op = Seq(
                OPS[primitive](C, stride),
                DropPath(drop_path) if drop_path else None,
            )
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(darts.Cell):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, drop_path):
        super().__init__(
            steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev,
            partial(MixedOp, drop_path=drop_path),
        )


class Network(darts.Network):
    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, drop_path=0, num_classes=10):
        self._drop_path = drop_path
        super().__init__(
            C, layers, steps, multiplier, stem_multiplier, num_classes,
            partial(Cell, drop_path=drop_path)
        )
