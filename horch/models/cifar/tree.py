import torch
import torch.nn as nn
import torch.nn.functional as F

from horch.models.modules import Conv2d, get_activation, get_norm_layer

class TreeCellA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
