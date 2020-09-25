import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.nas.darts.search.darts import parse_weights
from horch.nn import GlobalAvgPool
from horch.models.layers import Norm, Conv2d, Act

from horch.nas.operations import ReLUConvBN
from horch.nas.darts.genotypes import Genotype
from horch.nas.primitives import get_primitives
from horch.nas.darts.search.gdas import Cell, gumbel_sample


class NormalCell(Cell):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction_prev):
        super().__init__(
            steps, multiplier, C_prev_prev, C_prev, C, False, reduction_prev)


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


class Network(nn.Module):

    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, num_classes=10):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._steps = steps
        self._multiplier = multiplier
        self.tau = None

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
            if reduction:
                cell = ReductionCell(C_prev_prev, C_prev, C_curr)
            else:
                cell = NormalCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

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

        gene_normal = parse_weights(F.softmax(self.alphas.detach().cpu(), dim=0).numpy(), self._steps)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
        )
        return genotype
