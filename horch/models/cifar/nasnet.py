from toolz.curried import concat

import torch
import torch.nn as nn

from horch.nn import DropPath, GlobalAvgPool
from horch.models.layers import Act, Pool2d, Conv2d, Linear

from horch.nas.nasnet.genotypes import Genotype
from horch.nas.operations import FactorizedReduce, ReLUConvBN, OPS


def standardize(genotype: Genotype):
    if len(genotype.normal[0]) == 2:
        assert all(len(c) == 2 for c in genotype.normal)
        n = len(genotype.normal) // 2
        op_indices = concat([(i, i) for i in range(2, 2 + n)])
        op_names, indices = zip(*genotype.normal)
        normal = list(zip(op_names, op_indices, indices))

        assert all(len(c) == 2 for c in genotype.reduce)
        n = len(genotype.reduce) // 2
        op_indices = concat([(i, i) for i in range(2, 2 + n)])
        op_names, indices = zip(*genotype.reduce)
        reduce = list(zip(op_names, op_indices, indices))
    else:
        normal = genotype.normal
        reduce = genotype.reduce
    normal = sorted(normal, key=lambda c: (c[1], c[2]))
    reduce = sorted(reduce, key=lambda c: (c[1], c[2]))
    return Genotype(
            normal=normal, normal_concat=genotype.normal_concat,
            reduce=reduce, reduce_concat=genotype.reduce_concat)


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, drop_path):
        super().__init__()
        genotype = standardize(genotype)
        self.drop_path = drop_path
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1)

        if reduction:
            op_names, op_indices, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, op_indices, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, op_indices, indices, concat, reduction)

    def _compile(self, C, op_names, op_indices, indices, concat, reduction):
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        self._indices = []
        prev_op_index = 1
        for name, op_index, index in zip(op_names, op_indices, indices):
            if op_index != prev_op_index:
                self._ops.append(nn.ModuleList())
                self._indices.append([])

            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride)

            if self.drop_path and not isinstance(op, nn.Identity):
                op = nn.Sequential(
                    op,
                    DropPath(self.drop_path),
                )

            self._ops[-1].append(op)
            self._indices[-1].append(index)
            prev_op_index = op_index

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for ops, indices in zip(self._ops, self._indices):
            s = sum([op(states[index]) for op, index in zip(ops, indices)])
            states.append(s)
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super().__init__()
        self.features = nn.Sequential(
            Act('relu', inplace=True),
            Pool2d(5, stride=3, padding=0, type='avg'),
            Conv2d(C, 128, 1, norm='def', act='relu'),
            Conv2d(128, 768, 2, norm='def', act='relu', padding=0),
        )
        self.classifier = nn.Sequential(
            GlobalAvgPool(),
            Linear(768, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class NASNet(nn.Module):

    def __init__(self, C, layers, auxiliary, drop_path, num_classes, genotype):
        super().__init__()
        self._num_layers = layers
        self._auxiliary = auxiliary
        self._drop_path = drop_path

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = Conv2d(3, C_curr, 3, norm='def')

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, drop_path)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if auxiliary and i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.classifier = nn.Sequential(
            GlobalAvgPool(),
            Linear(C_prev, num_classes),
        )

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if self._auxiliary and i == 2 * self._num_layers // 3:
                logits_aux = self.auxiliary_head(s1)
        logits = self.classifier(s1)
        if self._auxiliary and self.training:
            return logits, logits_aux
        else:
            return logits
