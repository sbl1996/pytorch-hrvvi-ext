import random
import numpy as np

import torch

from horch.common import CUDA
from horch.train.classification.trainer import Trainer


def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)