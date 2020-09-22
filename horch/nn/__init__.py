import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.nn.max_unpool import MaxUnpool2d
from horch.nn.act import HardSwish, HardSigmoid, Swish
from horch.nn.drop import DropPath
from horch.nn.loss import CrossEntropyLoss


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.zeros(self.n_channels, dtype=torch.float32), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class GlobalAvgPool(nn.Module):

    def __init__(self, keep_dim=False):
        super().__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        if not self.keep_dim:
            x = x.view(x.size(0), -1)
        return x