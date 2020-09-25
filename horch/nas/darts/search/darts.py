from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.nn import DropPath, GlobalAvgPool
from horch.models.layers import Conv2d

from horch.nas.operations import OPS, FactorizedReduce, ReLUConvBN
from horch.nas.darts.genotypes import Genotype
from horch.nas.primitives import get_primitives


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in get_primitives():
            op = OPS[primitive](C, stride)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, op_cls=MixedOp):
        super().__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = op_cls(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, num_classes=10, cell_cls=Cell):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = Conv2d(3, C_curr, kernel_size=3, norm='def')

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = cell_cls(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.avg_pool = GlobalAvgPool()
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(2 + i for i in range(self._steps))
        num_ops = len(get_primitives())

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops), requires_grad=True)
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops), requires_grad=True)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.avg_pool(s1)
        logits = self.classifier(out)
        return logits

    def arch_parameters(self):
        return [self.alphas_normal, self.alphas_reduce]

    def model_parameters(self):
        ids = set(id(p) for p in self.arch_parameters())
        for p in self.parameters():
            if id(p) not in ids:
                yield p

    def genotype(self):

        gene_normal = parse_weights(F.softmax(self.alphas_normal.detach().cpu(), dim=0).numpy(), self._steps)
        gene_reduce = parse_weights(F.softmax(self.alphas_reduce.detach().cpu(), dim=0).numpy(), self._steps)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype


def parse_weights(weights, steps):
    PRIMITIVES = get_primitives()

    def get_op(w):
        if 'none' in PRIMITIVES:
            i = max([k for k in range(len(PRIMITIVES)) if k != PRIMITIVES.index('none')], key=lambda k: w[k])
        else:
            i = max(range(len(PRIMITIVES)), key=lambda k: w[k])
        return w[i], PRIMITIVES[i]

    gene = []
    start = 0
    for i in range(steps):
        end = start + i + 2
        W = weights[start:end]
        edges = sorted(range(i + 2), key=lambda x: -get_op(W[x])[0])[:2]
        for j in edges:
            gene.append((get_op(W[j])[1], j))
        start = end
    return gene