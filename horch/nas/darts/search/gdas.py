import torch
import torch.nn.functional as F
from horch.nas.darts.search import darts
from horch.train.v3.callbacks import Callback


def gumbel_sample(a, tau):
    hardwts = F.gumbel_softmax(a, tau, hard=True)
    indices = hardwts.argmax(dim=1).tolist()
    return hardwts, indices


class MixedOp(darts.MixedOp):

    def forward(self, x, hardwts, index):
        return sum(hardwts[i] * op(x) if i == index else hardwts[i] for i, op in enumerate(self._ops))


class Cell(darts.Cell):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__(
            steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, MixedOp)

    def forward(self, s0, s1, hardwts, indices):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, hardwts[offset + j], int(indices[offset + j])) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(darts.Network):

    def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3, num_classes=10):
        super().__init__(
            C, layers, steps, multiplier, stem_multiplier, num_classes, Cell
        )

    def forward(self, x):
        s0 = s1 = self.stem(x)
        hardwts_reduce, indices_reduce = gumbel_sample(self.alphas_reduce, self.tau)
        hardwts_normal, indices_normal = gumbel_sample(self.alphas_normal, self.tau)
        for cell in self.cells:
            hardwts = hardwts_reduce if cell.reduction else hardwts_normal
            indices = indices_reduce if cell.reduction else indices_normal
            s0, s1 = s1, cell(s0, s1, hardwts, indices)
        out = self.avg_pool(s1)
        logits = self.classifier(out)
        return logits


class TauSchedule(Callback):

    def __init__(self, tau_max, tau_min):
        super().__init__()
        self.tau_max, self.tau_min = tau_max, tau_min

    def begin_epoch(self, state):
        tau_max = self.tau_max
        tau_min = self.tau_min
        tau = tau_max - (tau_max - tau_min) * (state['epoch'] / state['epochs'])
        self.learner.model.tau = tau
