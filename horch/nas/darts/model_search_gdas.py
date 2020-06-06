import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import get_norm_layer, Conv2d, get_activation

from horch.nas.darts.operations import OPS, FactorizedReduce, ReLUConvBN
from horch.nas.darts.genotypes import PRIMITIVES, Genotype


class MixedOp(nn.Module):

    def __init__(self, C):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, 1)
            if 'pool' in primitive:
                op = nn.Sequential(op, get_norm_layer(C))
            self._ops.append(op)

    def forward(self, x, hardwts, index):
        return sum(hardwts[i] * op(x) if i == index else hardwts[i] for i, op in enumerate(self._ops))


class NormalCell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction_prev):
        super().__init__()

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
                op = MixedOp(C)
                ops.append(op)
            self._ops.append(ops)

    def forward(self, s0, s1, hardwts, indices):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            s = sum(self._ops[i][j](h, hardwts[i][j], int(indices[i][j])) for j, h in enumerate(states))
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class ReductionCell(nn.Module):

    def __init__(self, C_prev_prev, C_prev, C):
        super().__init__()
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1)

        self.branch_a1 = nn.Sequential(
            get_activation(),
            Conv2d(C, C, (1, 3), stride=(1, 2), groups=8, bias=False),
            Conv2d(C, C, (3, 1), stride=(2, 1), groups=8, bias=False),
            get_norm_layer(C, affine=True),
            get_activation(),
            Conv2d(C, C, 1),
            get_norm_layer(C, affine=True),
        )
        self.branch_a2 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            get_norm_layer(C, affine=True)
        )
        self.branch_a1 = nn.Sequential(
            get_activation(),
            Conv2d(C, C, (1, 3), stride=(1, 2), groups=8, bias=False),
            Conv2d(C, C, (3, 1), stride=(2, 1), groups=8, bias=False),
            get_norm_layer(C, affine=True),
            get_activation(),
            Conv2d(C, C, 1),
            get_norm_layer(C, affine=True),
        )
        self.branch_a2 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            get_norm_layer(C, affine=True)
        )

    def forward(self, s0, s1, *args):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        x0 = self.branch_a1(s0)
        x1 = self.branch_a1(s1)
        # if self.training and drop_prob > 0.:
        #     X0, X1 = drop_path(X0, drop_prob), drop_path(X1, drop_prob)

        x2 = self.branch_a2(s0)
        x3 = self.branch_a1(s1)
        # if self.training and drop_prob > 0.:
        #     X2, X3 = drop_path(X2, drop_prob), drop_path(X3, drop_prob)

        return torch.cat([x0, x1, x2, x3], dim=1)


def gumbel_sample(a, tau):
    while True:
        gumbels = -torch.empty_like(a).exponential_().log()
        logits = (a.log_softmax(dim=1) + gumbels) / tau
        probs = nn.functional.softmax(logits, dim=1)
        index = probs.max(-1, keepdim=True)[1]
        one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        hardwts = one_h - probs.detach() + probs
        if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
            continue
        else:
            return hardwts, [int(i) for i in index.cpu().numpy()]


class Network(nn.Module):

    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, num_classes=10, tau=10.0):
        super().__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self.steps = steps
        self.multiplier = multiplier
        self.tau = tau

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
            if reduction:
                cell = ReductionCell(C_prev_prev, C_prev, C_curr)
            else:
                cell = NormalCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, x):
        hardwts, indices = zip(*[ gumbel_sample(a, self.tau) for a in self.alphas ])
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, hardwts, indices)
        out = self.avg_pool(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)

        self.alphas = nn.ParameterList([
            nn.Parameter(1e-3 * torch.randn(i + 2, num_ops), requires_grad=True)
            for i in range(self.steps)
        ])

    def model_parameters(self):
        ids = set(id(p) for p in self.arch_parameters())
        for p in self.parameters():
            if id(p) not in ids:
                yield p

    def arch_parameters(self):
        return self.alphas.parameters()

    def genotype(self):

        def _parse(weights):
            gene = []
            for i in range(self.steps):
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

        gene_normal = _parse([F.softmax(w.detach().cpu(), dim=0).numpy() for w in self.alphas])

        concat = range(2 + self.steps - self.multiplier, self.steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
        )
        return genotype
