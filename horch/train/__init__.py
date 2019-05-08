from horch.train.trainer import Trainer
from horch.train.gan_trainer import GANTrainer

from toolz import curry
import torch.nn as nn


@curry
def init_weights(m, a=0, nonlinearity='leaky_relu', mode='fan_in'):
    name = type(m).__name__
    if name.find("Linear") != -1 or name.find("Conv") != -1:
        if m.weight.requires_grad:
            nn.init.kaiming_uniform_(
                m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
