from collections import defaultdict


class AnchorGeneratorBase:

    def __init__(self, cache=True):
        self.cache = cache
        self.caches = defaultdict(dict)

    def calculate(self, grid_sizes, device, dtype):
        raise NotImplementedError

    def __call__(self, grid_sizes, device, dtype):
        if not isinstance(grid_sizes, tuple):
            grid_sizes = tuple(grid_sizes)
        if self.cache:
            if (device, dtype) in self.caches:
                caches = self.caches[(device, dtype)]
                if grid_sizes not in caches:
                    caches[grid_sizes] = self.calculate(grid_sizes, device, dtype)
            else:
                caches = {
                    grid_sizes: self.calculate(grid_sizes, device, dtype)
                }
                self.caches[(device, dtype)] = caches
            return caches[grid_sizes]
        else:
            return self.calculate(grid_sizes, device, dtype)