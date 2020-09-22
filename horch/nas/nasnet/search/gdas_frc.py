import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.layers import Norm, Conv2d, Act

from horch.nas.operations import OPS, FactorizedReduce, ReLUConvBN
from horch.nas.nasnet.genotypes import get_primitives, Genotype
from horch.nn import GlobalAvgPool


class MixedOp(nn.Module):

    def __init__(self, C):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in get_primitives():
            op = OPS[primitive](C, 1)
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
            for j in range(2 + i):
                op = MixedOp(C)
                self._ops.append(op)

    def forward(self, s0, s1, hardwts, indices):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, hardwts[offset + j], int(indices[offset + j])) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class ReductionCell(nn.Module):

    def __init__(self, C_prev_prev, C_prev, C):
        super().__init__()
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1)

        self.branch_a1 = nn.Sequential(
            Act(),
            Conv2d(C, C, (1, 3), stride=(1, 2), groups=8, bias=False),
            Conv2d(C, C, (3, 1), stride=(2, 1), groups=8, bias=False),
            Norm(C, affine=True),
            Act(),
            Conv2d(C, C, 1),
            Norm(C, affine=True),
        )
        self.branch_a2 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            Norm(C, affine=True),
        )
        self.branch_b1 = nn.Sequential(
            Act(),
            Conv2d(C, C, (1, 3), stride=(1, 2), groups=8, bias=False),
            Conv2d(C, C, (3, 1), stride=(2, 1), groups=8, bias=False),
            Norm(C, affine=True),
            Act(),
            Conv2d(C, C, 1),
            Norm(C, affine=True),
        )
        self.branch_b2 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            Norm(C, affine=True),
        )

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        x0 = self.branch_a1(s0)
        x1 = self.branch_a2(s1)

        x2 = self.branch_b1(s0)
        x3 = self.branch_b2(s1)

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
            Norm(C_curr),
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

        self.post_activ = nn.Sequential(
            Norm(C_prev),
            Act(),
        )
        self.avg_pool = GlobalAvgPool()
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(2 + i for i in range(self._steps))
        num_ops = len(get_primitives())

        self.alphas = nn.Parameter(1e-3 * torch.randn(k, num_ops), requires_grad=True)

    def forward(self, x):
        hardwts, indices = gumbel_sample(self.alphas, self.tau)
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, hardwts, indices)
        s1 = self.post_activ(s1)
        out = self.avg_pool(s1)
        logits = self.classifier(out)
        return logits

    def arch_parameters(self):
        return [self.alphas]

    def model_parameters(self):
        ids = set(id(p) for p in self.arch_parameters())
        for p in self.parameters():
            if id(p) not in ids:
                yield p

    def genotype(self):
        PRIMITIVES = get_primitives()

        def get_op(w):
            if 'none' in PRIMITIVES:
                i = max([k for k in range(len(PRIMITIVES)) if k != PRIMITIVES.index('none')], key=lambda k: w[k])
            else:
                i = max(range(len(PRIMITIVES)), key=lambda k: w[k])
            return w[i], PRIMITIVES[i]

        def _parse(weights):
            gene = []
            start = 0
            for i in range(self._steps):
                end = start + i + 2
                W = weights[start:end]
                edges = sorted(range(i + 2), key=lambda x: -get_op(W[x])[0])[:2]
                for j in edges:
                    gene.append((get_op(W[j])[1], j))
                start = end
            return gene

        gene_normal = _parse(F.softmax(self.alphas.detach().cpu(), dim=0).numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
        )
        return genotype
