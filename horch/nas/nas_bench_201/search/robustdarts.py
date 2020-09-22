from functools import partial

import torch.nn as nn

from horch.nn import DropPath
from horch.models.layers import Seq

from horch.nas.operations import OPS
from horch.nas.primitives import PRIMITIVES_nas_bench_201
from horch.nas.nas_bench_201.search import darts


class MixedOp(nn.Module):

    def __init__(self, C, drop_path=0):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_nas_bench_201:
            op = Seq(
                OPS[primitive](C, 1),
                DropPath(drop_path) if drop_path else None,
            )
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(darts.Cell):

    def __init__(self, steps, C_prev, C, drop_path):
        super().__init__(
            steps, C_prev, C,
            partial(MixedOp, drop_path=drop_path),
        )


class Network(darts.Network):

    def __init__(self, C, layers, steps=3, stem_multiplier=3, drop_path=0.6, num_classes=10):
        super().__init__(C, layers, steps, stem_multiplier, num_classes,
                         partial(Cell, drop_path=drop_path))
