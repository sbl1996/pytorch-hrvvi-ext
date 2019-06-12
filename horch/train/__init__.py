from typing import Sequence

import torch
from horch import ProtectedSeq
from horch.train.trainer import Trainer, ValSet
from horch.train.gan_trainer import GANTrainer
from horch.transforms.detection import BoxList

from toolz import curry
import torch.nn as nn
from torch.utils.data.dataloader import default_collate


@curry
def init_weights(m, a=0, nonlinearity='leaky_relu', mode='fan_in'):
    name = type(m).__name__
    if name.find("Linear") != -1 or name.find("Conv") != -1:
        if m.weight.requires_grad:
            nn.init.kaiming_uniform_(
                m.weight, a=a, mode=mode, nonlinearity=nonlinearity)


def misc_collate(batch):
    input, target = zip(*batch)
    if torch.is_tensor(input[0]):
        input = default_collate(input)
    else:
        if any([torch.is_tensor(t) for t in input[0]]):
            input = [default_collate(t) if torch.is_tensor(t[0]) else t for t in zip(*input)]
        else:
            input = ProtectedSeq(input)
    if torch.is_tensor(target[0]):
        target = default_collate(target)
    elif isinstance(target[0], BoxList):
        target = ProtectedSeq(target)
    elif isinstance(target[0], Sequence):
        if len(target[0]) == 0:
            target = []
        else:
            if any([torch.is_tensor(t) for t in target[0]]):
                target = [default_collate(t) if torch.is_tensor(t[0]) else t for t in zip(*target)]
            else:
                target = ProtectedSeq(target)
    else:
        target = ProtectedSeq(target)
    return input, target