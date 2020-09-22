import math

from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, epochs, min_lr=0, warmup_epoch=0, warmup_min_lr=None, last_epoch=-1):
        self.epochs = epochs
        self.min_lr = min_lr
        self.warmup_epoch = warmup_epoch
        self.warmup_min_lr = warmup_min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch == 0:
            if self.warmup_epoch == 0:
                return self.base_lrs
            return [self.warmup_min_lr for _ in self.base_lrs]
        elif epoch < self.warmup_epoch:
            min_lr = self.warmup_min_lr
            return [
                epoch / self.warmup_epoch * (base_lr - min_lr) + min_lr
                for base_lr in self.base_lrs
            ]
        elif self.warmup_epoch <= epoch < self.epochs:
            epoch = epoch - self.warmup_epoch
            epochs = self.epochs - self.warmup_epoch
            factor = (1 + math.cos(math.pi * epoch / epochs)) / 2
            return [
                factor * (base_lr - self.min_lr) + self.min_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [self.min_lr for _ in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


CosineLR = CosineAnnealingLR
