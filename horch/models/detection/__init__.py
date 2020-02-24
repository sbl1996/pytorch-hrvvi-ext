import torch
import torch.nn as nn

from horch.models.modules import Sequential
from horch.models.detection.ssd import SSD
from horch.models.detection.retinanet import RetinaNet

from horch.common import tuplify


def split_levels(levels, split_at=5):
    levels = tuplify(levels)
    lo = levels[0]
    hi = levels[-1]
    assert levels == tuple(range(lo, hi + 1))
    basic_levels = tuple(range(lo, min(hi, split_at) + 1))
    extra_levels = tuple(range(max(lo, split_at + 1), hi + 1))
    return basic_levels, extra_levels
