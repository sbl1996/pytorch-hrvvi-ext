import torch.nn.functional as F
from horch.nas.darts.search.gdas import TauSchedule, gumbel_sample
from horch.nas.nas_bench_201.search import darts


class MixedOp(darts.MixedOp):

    def forward(self, x, hardwts, index):
        return sum(hardwts[i] * op(x) if i == index else hardwts[i] for i, op in enumerate(self._ops))


class Cell(darts.Cell):

    def __init__(self, steps, C_prev, C):
        super().__init__(steps, C_prev, C, MixedOp)

    def forward(self, s, hardwts, indices):
        s = self.preprocess(s)

        states = [s]
        offset = 0
        for i in range(self._steps):
            s = sum(
                self._ops[offset + j](h, hardwts[offset + j], int(indices[offset + j])) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return states[-1]


class Network(darts.Network):

    def __init__(self, C, layers, steps=3, stem_multiplier=3, num_classes=10):
        super().__init__(C, layers, steps, stem_multiplier, num_classes, Cell)

    def forward(self, x):
        s = self.stem(x)
        hardwts, indices = gumbel_sample(self.alphas, self.tau)
        for cell in self.cells:
            s = cell(s, hardwts, indices)
        x = self.avg_pool(s)
        logits = self.classifier(x)
        return logits