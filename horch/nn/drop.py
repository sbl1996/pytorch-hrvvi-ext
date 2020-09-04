import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        assert 0 <= p <= 1, "drop probability has to be between 0 and 1, but got %f" % p
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep_prob = 1.0 - self.p
        batch_size = x.size(0)
        t = torch.rand(batch_size, 1, 1, 1, dtype=x.dtype, device=x.device) < keep_prob
        x = (x / keep_prob).masked_fill(t, 0)
        return x

    def extra_repr(self):
        return 'p={}'.format(self.p)