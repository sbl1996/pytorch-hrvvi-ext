from collections import defaultdict
from bidict import bidict

import torch
from horch.detection import calc_grid_sizes


class AnchorGeneratorBase:

    def __init__(self, levels=(3, 4, 5), cache=True):
        self.levels = levels
        self.strides = [2 ** l for l in levels]
        self.cache = cache
        self.caches = defaultdict(dict)
        self.size2grid_sizes = bidict()

    def register_size(self, size, grid_sizes=None):
        if not isinstance(size, tuple):
            size = tuple(size)
        if grid_sizes is None:
            grid_sizes = calc_grid_sizes(size, self.strides)
        self.size2grid_sizes[size] = grid_sizes

    def calculate(self, size, grid_sizes, device, dtype):
        raise NotImplementedError

    def __call__(self, size, device, dtype):
        # size: (w, h)
        if len(size) == len(self.levels) and isinstance(size[0], torch.Size):
            grid_sizes = tuple(size)
            size = self.size2grid_sizes.inv[grid_sizes]
        elif isinstance(size, tuple) and len(size) == 2:
            grid_sizes = self.size2grid_sizes[size]
        else:
            raise ValueError("Invalid size.")
        if self.cache:
            if (device, dtype) in self.caches:
                caches = self.caches[(device, dtype)]
                if size not in caches:
                    caches[size] = self.calculate(size, grid_sizes, device, dtype)
            else:
                caches = {
                    size: self.calculate(size, grid_sizes, device, dtype)
                }
                self.caches[(device, dtype)] = caches
            return caches[size]
        else:
            return self.calculate(size, grid_sizes, device, dtype)