from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import get_norm_layer, Conv2d

from horch.nas.darts.operations import OPS, FactorizedReduce, ReLUConvBN
from horch.nas.darts.genotypes import PRIMITIVES, Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride)
            if 'pool' in primitive:
                op = nn.Sequential(op, get_norm_layer(C))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
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
            ops = nn.ModuleList()
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                ops.append(op)
            self._ops.append(ops)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            s = sum(self._ops[i][j](h, weights[i][j]) for j, h in enumerate(states))
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, num_classes=10):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            Conv2d(3, C_curr, kernel_size=3, bias=False),
            get_norm_layer(C_curr),
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for cell in self.cells:
            weights = self.alphas_reduce if cell.reduction else self.alphas_normal
            weights = [F.softmax(w, dim=0) for w in weights]
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.avg_pool(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.ParameterList([
            nn.Parameter(1e-3 * torch.randn(i + 2, num_ops), requires_grad=True)
            for i in range(self._steps)
        ])
        self.alphas_reduce = nn.ParameterList([
            nn.Parameter(1e-3 * torch.randn(i + 2, num_ops), requires_grad=True)
            for i in range(self._steps)
        ])

    def model_parameters(self):
        ids = set(id(p) for p in self.arch_parameters())
        for p in self.parameters():
            if id(p) not in ids:
                yield p

    def arch_parameters(self):
        return chain(self.alphas_normal.parameters(), self.alphas_reduce.parameters())

    def genotype(self):

        def _parse(weights):
            gene = []
            for i in range(self._steps):
                W = weights[i].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
            return gene

        gene_normal = _parse([F.softmax(w.detach().cpu(), dim=0).numpy() for w in self.alphas_normal])
        gene_reduce = _parse([F.softmax(w.detach().cpu(), dim=0).numpy() for w in self.alphas_reduce])

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
