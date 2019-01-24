from hutil.train.trainer import Trainer
from hutil.train.gan_trainer import GANTrainer
from hutil.train.metrics import Loss, LossD, LossG, Accuracy

from toolz import curry
import torch.nn as nn


@curry
def init_weights(m, nonlinearity='leaky_relu', mode='fan_in'):
    typ = type(m)
    if typ == nn.Linear or typ == nn.Conv2d:
        nn.init.kaiming_uniform_(
            m.weight, mode=mode, nonlinearity=nonlinearity)
