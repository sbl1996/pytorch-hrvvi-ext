import random
import numpy as np

from toolz import curry

import torch
import torch.nn as nn

from horch.common import CUDA
# from horch.train.gan import GANTrainer
from horch.train.trainer import Trainer


@curry
def init_weights(m, a=0, nonlinearity='leaky_relu', mode='fan_in'):
    name = type(m).__name__
    if name.find("Linear") != -1 or name.find("Conv") != -1:
        if m.weight.requires_grad:
            nn.init.kaiming_uniform_(
                m.weight, a=a, mode=mode, nonlinearity=nonlinearity)


def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)