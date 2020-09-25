from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.nas.operations import OPS
from horch.nas.darts.genotypes import Genotype
from horch.nas.primitives import get_primitives
import horch.nas.darts.search.darts as darts


def channel_shuffle(x, g):
    b, c, h, w = x.size()

    x = x.view(b, g, c // g, h, w).permute(
        0, 2, 1, 3, 4).reshape(b, c, h, w)
    return x


class MixedOp(nn.Module):

    def __init__(self, C, stride, k):
        super().__init__()
        self.stride = stride
        self.mp = nn.MaxPool2d(2, 2)
        self.k = k
        self._channels = C // k

        self._ops = nn.ModuleList()
        for primitive in get_primitives():
            op = OPS[primitive](self._channels, stride)
            self._ops.append(op)

    def forward(self, x, weights):
        x1 = x[:, :self._channels, :, :]
        x2 = x[:, self._channels:, :, :]
        x1 = sum(w * op(x1) for w, op in zip(weights, self._ops))
        if self.stride == 1:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = torch.cat([x1, self.mp(x2)], dim=1)
        x = channel_shuffle(x, self.k)
        return x


class Cell(darts.Cell):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, k):
        super().__init__(
            steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev,
            partial(MixedOp, k=k)
        )

    def forward(self, s0, s1, alphas, betas):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum([betas[offset + j] * self._ops[offset + j](h, alphas[offset + j]) for j, h in enumerate(states)])
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)

def beta_softmax(betas, steps, scale=False):
    beta_list = []
    offset = 0
    for i in range(steps):
        beta = F.softmax(betas[offset:(offset + i + 2)], dim=0)
        if scale:
            beta = beta * len(beta)
        beta_list.append(beta)
        offset += i + 2
    betas = torch.cat(beta_list, dim=0)
    return betas


class Network(darts.Network):

    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, k=4, num_classes=10):
        self._k = k
        super().__init__(
            C, layers, steps, multiplier, stem_multiplier, num_classes,
            partial(Cell, k=k)
        )

    def _initialize_alphas(self):
        k = sum(2 + i for i in range(self._steps))
        num_ops = len(get_primitives())

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops), requires_grad=True)
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops), requires_grad=True)

        self.betas_normal = nn.Parameter(1e-3 * torch.randn(k), requires_grad=True)
        self.betas_reduce = nn.Parameter(1e-3 * torch.randn(k), requires_grad=True)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        alphas_reduce = F.softmax(self.alphas_reduce, dim=-1)
        alphas_normal = F.softmax(self.alphas_normal, dim=-1)

        betas_reduce = beta_softmax(self.betas_reduce, self._steps)
        betas_normal = beta_softmax(self.betas_normal, self._steps)

        for cell in self.cells:
            alphas = alphas_reduce if cell.reduction else alphas_normal
            betas = betas_reduce if cell.reduction else betas_normal
            s0, s1 = s1, cell(s0, s1, alphas, betas)
        out = self.avg_pool(s1)
        logits = self.classifier(out)
        return logits

    def arch_parameters(self):
        return [self.alphas_normal, self.alphas_reduce, self.betas_normal, self.betas_reduce]

    def genotype(self):
        alphas_normal = F.softmax(self.alphas_normal.detach().cpu(), dim=0).numpy()
        betas_normal = beta_softmax(self.betas_normal.detach().cpu(), self._steps).numpy()
        alphas_normal = alphas_normal * betas_normal[:, None]

        alphas_reduce = F.softmax(self.alphas_reduce.detach().cpu(), dim=0).numpy()
        betas_reduce = beta_softmax(self.betas_reduce.detach().cpu(), self._steps).numpy()
        alphas_reduce = alphas_reduce * betas_reduce[:, None]

        gene_normal = darts.parse_weights(alphas_normal, self._steps)
        gene_reduce = darts.parse_weights(alphas_reduce, self._steps)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
