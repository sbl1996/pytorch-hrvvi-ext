import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.nn import GlobalAvgPool
from horch.models.layers import Conv2d, Act

from horch.nas.primitives import PRIMITIVES_nas_bench_201
from horch.nas.operations import OPS, ReLUConvBN


class MixedOp(nn.Module):

    def __init__(self, C):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES_nas_bench_201:
            op = OPS[primitive](C, 1)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, C_prev, C, op_cls=MixedOp):
        super().__init__()

        self.preprocess = ReLUConvBN(C_prev, C, 1, 1)
        self._steps = steps
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(1 + i):
                op = op_cls(C)
                self._ops.append(op)

    def forward(self, s, weights):
        s = self.preprocess(s)

        states = [s]
        offset = 0
        for i in range(self._steps):
            s = sum([self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states)])
            offset += len(states)
            states.append(s)

        return states[-1]


class BasicBlock(nn.Module):

    def __init__(self, C_prev, C, stride=2):
        super().__init__()
        assert stride == 2
        self.conv1 = ReLUConvBN(C_prev, C, 3, stride=stride)
        self.conv2 = ReLUConvBN(C, C, 3, stride=1)
        self.downsample = nn.Sequential(
            nn.AvgPool2d(2, 2, count_include_pad=False),
            Conv2d(C_prev, C, 1),
        )
        self.act = Act()
        self.stride = stride

    def forward(self, x, *args):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.downsample(identity)
        x = self.act(x)
        return x


class Network(nn.Module):

    def __init__(self, C, layers, steps=3, stem_multiplier=3, num_classes=10, cell_cls=Cell):
        super().__init__()
        self._C = C
        self._steps = steps

        C_curr = stem_multiplier * C
        self.stem = Conv2d(3, C_curr, 3, norm='default')

        C_prev, C_curr = C_curr, C
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                cell = BasicBlock(C_prev, C_curr, stride=2)
            else:
                cell = cell_cls(steps, C_prev, C_curr)
            self.cells.append(cell)
            C_prev = C_curr

        self.avg_pool = GlobalAvgPool()
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 + i for i in range(self._steps))
        num_ops = len(PRIMITIVES_nas_bench_201)

        self.alphas = nn.Parameter(1e-3 * torch.randn(k, num_ops), requires_grad=True)

    def forward(self, x):
        s = self.stem(x)
        weights = F.softmax(self.alphas, dim=-1)
        for cell in self.cells:
            s = cell(s, weights)
        x = self.avg_pool(s)
        logits = self.classifier(x)
        return logits

    def arch_parameters(self):
        return [self.alphas]

    def model_parameters(self):
        ids = set(id(p) for p in self.arch_parameters())
        for p in self.parameters():
            if id(p) not in ids:
                yield p

    def genotype(self):
        PRIMITIVES = PRIMITIVES_nas_bench_201

        def get_op(w):
            if 'none' in PRIMITIVES:
                i = max([k for k in range(len(PRIMITIVES)) if k != PRIMITIVES.index('none')], key=lambda k: w[k])
            else:
                i = max(range(len(PRIMITIVES)), key=lambda k: w[k])
            return PRIMITIVES[i]

        def _parse(weights):
            genes = []
            start = 0
            for i in range(self._steps):
                gene = []
                end = start + i + 1
                W = weights[start:end]
                for j in range(i + 1):
                    gene.append((get_op(W[j]), j))
                start = end
                genes.append(gene)
            return genes

        gene = _parse(F.softmax(self.alphas, dim=-1).cpu().detach().numpy())
        s = "+".join([f"|{'|'.join(f'{op}~{i}' for op, i in ops)}|" for ops in gene])
        return s
