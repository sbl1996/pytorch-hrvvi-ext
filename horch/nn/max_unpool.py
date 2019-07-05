import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxUnpool2d(nn.Module):
    def __init__(self, cache=False):
        super().__init__()
        self.cache = cache
        if self.cache:
            self.indices = None

    def compute_indices(self, x):
        b, c, h, w = x.size()
        oh, ow = 2 * h, 2 * w
        x = -torch.arange(oh * ow, dtype=x.dtype, device=x.device).view(1, 1, oh, ow)
        indices = F.max_pool2d_with_indices(x, (2, 2), (2, 2))[1].expand(b, c, h, w)
        return indices

    def forward(self, x):
        if self.cache:
            if self.indices is None:
                self.indices = self.compute_indices(x)
            indices = self.indices
        else:
            indices = self.compute_indices(x)
        return F.max_unpool2d(x, indices, (2, 2), (2, 2))
