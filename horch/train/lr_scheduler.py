import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{i}}\pi))

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0`(after restart), set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of epochs for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        iters_per_epoch (int, optional):
            Number of iterations per epoch. If provided, it will be used to set
            weight decay dynamically as :math: `\lambda = \lambda_{norm}\sqrt{\frac{b}{BT}}`.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, warmup=0, last_epoch=-1, iters_per_epoch=None):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup = warmup
        self.iters_per_epoch = iters_per_epoch
        super().__init__(optimizer, last_epoch)
        self.T_cur = last_epoch

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch < self.warmup:
                eta_min = base_lr * 0.1
                # T_cur = self.last_epoch
                T_cur = self.T_cur + self.warmup
                T_i = self.warmup
                mult = (1 + math.cos(math.pi * (1 + T_cur / T_i))) / 2
            else:
                eta_min = self.eta_min
                T_cur = self.T_cur
                T_i = self.T_i
                mult = (1 + math.cos(math.pi * T_cur / T_i)) / 2
            lr = eta_min + (base_lr - eta_min) * mult
            lrs.append(lr)
        return lrs

    def step(self, epoch=None):
        """Step could be called after every update, i.e. if one epoch has 10 iterations
        (number_of_train_examples / batch_size), we should call SGDR.step(0.1), SGDR.step(0.2), etc.

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            if epoch >= self.warmup:
                self.T_cur = self.T_cur + 1
                if self.T_cur >= self.T_i:
                    self.T_cur = self.T_cur - self.T_i
                    self.T_i = self.T_i * self.T_mult
            self.last_epoch = math.floor(epoch)
        else:
            epoch -= self.warmup
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * (self.T_mult ** n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
            self.last_epoch = math.floor(epoch + self.warmup)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            defaults = self.optimizer.defaults
            base_weight_decay = defaults['weight_decay']
            if self.iters_per_epoch:
                base_weight_decay *= math.sqrt(1 / (self.iters_per_epoch * self.T_i))
            param_group['weight_decay'] = lr / defaults['lr'] * base_weight_decay
