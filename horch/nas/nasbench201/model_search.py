import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import Norm, Conv2d
from horch.nas.nasbench201.operations import OPS, ReLUConvBN
from horch.nas.darts.genotypes import PRIMITIVES


class MixedOp(nn.Module):

    def __init__(self, primitives, C):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](C, 1)
            if 'pool' in primitive:
                op = nn.Sequential(op, Norm(C))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class ReductionCell(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch1 = nn.Sequential(
            ReLUConvBN(in_channels, out_channels, 3, stride=2),
            ReLUConvBN(out_channels, out_channels, 3)
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            ReLUConvBN(in_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.branch1(x) + self.branch2(x)


class NormalCell(nn.Module):

    def __init__(self, primitives, nodes, C):
        super().__init__()

        self.nodes = nodes

        self.ops = nn.ModuleList()
        for i in range(1, self.nodes):
            ops = nn.ModuleList()
            for j in range(nodes):
                op = MixedOp(primitives, C)
                ops.append(op)
            self.ops.append(ops)

    def forward(self, x, weights):
        states = [x]
        for i in range(self.nodes-1):
            s = sum(self.ops[i][j](h, weights[i][j]) for j, h in enumerate(states))
            states.append(s)
        return states[-1]


class Network(nn.Module):

    def __init__(self, primitives=PRIMITIVES, C=16, num_stacked=5, nodes=4, num_classes=10):
        super().__init__()
        self.primitives = primitives
        self.C = C
        self.num_classes = num_classes
        self.num_stacked = num_stacked
        self.nodes = nodes

        self.stem = Conv2d(3, C, kernel_size=3, norm='default')
        for i in range(3):
            if i != 0:
                self.add_module("reduce%d" % i, ReductionCell(C, C * 2))
                C = C * 2
            stage = nn.ModuleList()
            for _ in range(num_stacked):
                stage.append(NormalCell(primitives, nodes, C))
            self.add_module("stage%d" % (i + 1), stage)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, num_classes)

        self._initialize_alphas()

    def forward(self, x):
        x = self.stem(x)
        weights = [F.softmax(w, dim=0) for w in self.alphas]
        for cell in self.stage1:
            x = cell(x, weights)
        x = self.reduce1(x)
        for cell in self.stage2:
            x = cell(x, weights)
        x = self.reduce2(x)
        for cell in self.stage3:
            x = cell(x, weights)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)

        self.alphas = nn.ParameterList([
            nn.Parameter(1e-3 * torch.randn(i, num_ops), requires_grad=True)
            for i in range(1, self.nodes)
        ])

    def model_parameters(self):
        ids = set(id(p) for p in self.arch_parameters())
        for p in self.parameters():
            if id(p) not in ids:
                yield p

    def arch_parameters(self):
        return self.alphas.parameters()
