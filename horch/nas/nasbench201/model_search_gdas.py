import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import get_norm_layer, Conv2d, get_activation
from horch.nas.nasbench201.operations import OPS, ReLUConvBN
from horch.nas.darts.genotypes import PRIMITIVES


class MixedOp(nn.Module):

    def __init__(self, primitives, C):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in primitives:
            op = OPS[primitive](C, 1)
            if 'pool' in primitive:
                op = nn.Sequential(op, get_norm_layer(C))
            self._ops.append(op)

    def forward(self, x, hardwts, index):
        return sum(hardwts[i] * op(x) if i == index else hardwts[i] for i, op in enumerate(self._ops))


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

    def forward(self, x, hardwts, indices):
        states = [x]
        for i in range(self.nodes - 1):
            s = sum(self.ops[i][j](h, hardwts[i][j], int(indices[i][j])) for j, h in enumerate(states))
            states.append(s)
        return states[-1]


def gumbel_sample(a, tau):
    o = -torch.log(-torch.log(torch.rand(*a.size(), device=a.device)))
    return F.softmax((F.log_softmax(a, dim=1) + o) / tau, dim=1)



class Network(nn.Module):

    def __init__(self, primitives=PRIMITIVES, C=16, num_stacked=5, nodes=4, num_classes=10, tau=10.0):
        super().__init__()
        self.primitives = primitives
        self.C = C
        self.num_classes = num_classes
        self.num_stacked = num_stacked
        self.nodes = nodes
        self.tau = tau

        self.stem = Conv2d(3, C, kernel_size=3, norm_layer='default')
        for i in range(3):
            if i != 0:
                self.add_module("reduce%d" % i, ReductionCell(C, C * 2))
                C = C * 2
            stage = nn.ModuleList()
            for _ in range(num_stacked):
                stage.append(NormalCell(primitives, nodes, C))
            self.add_module("stage%d" % (i + 1), stage)

        self.post_activ = nn.Sequential(
            get_norm_layer(C),
            get_activation(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, num_classes)

        self._initialize_alphas()

    def forward(self, x):
        x = self.stem(x)
        weights = [gumbel_sample(w, self.tau) for w in self.alphas]
        indices = [torch.argmax(w, dim=1) for w in weights]
        one_h = [F.one_hot(i, len(self.primitives)) for i in indices]
        indices = [index.cpu().numpy() for index in indices]
        hardwts = [h - w.detach() + w for w, h in zip(weights, one_h)]
        for cell in self.stage1:
            x = cell(x, hardwts, indices)
        x = self.reduce1(x)
        for cell in self.stage2:
            x = cell(x, hardwts, indices)
        x = self.reduce2(x)
        for cell in self.stage3:
            x = cell(x, hardwts, indices)
        x = self.post_activ(x)
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
