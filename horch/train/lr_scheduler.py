import math

from torch.optim import Optimizer
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

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, warmup=0, warmup_eta_min=None, gamma=1.0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup = warmup
        self.warmup_eta_min = warmup_eta_min
        self.gamma = gamma
        self._gamma = 1.0
        super().__init__(optimizer, last_epoch)
        self.T_cur = last_epoch

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch < self.warmup:
                eta_min = base_lr * 0.1 if self.warmup_eta_min is None else self.warmup_eta_min
                # T_cur = self.last_epoch
                T_cur = self.T_cur + self.warmup
                T_i = self.warmup
                mult = (1 + math.cos(math.pi * (1 + T_cur / T_i))) / 2
            else:
                eta_min = self.eta_min
                T_cur = self.T_cur
                T_i = self.T_i
                mult = (1 + math.cos(math.pi * T_cur / T_i)) / 2
            mult *= self._gamma
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
                    self._gamma = self.gamma ** (epoch // self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * (self.T_mult ** n)
                    self._gamma = self.gamma ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
            self.last_epoch = math.floor(epoch + self.warmup)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CyclicLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        momentum_key (str): momentum for SGD or betas for Adam.
            Default: 'momentum'
        base_momentum (float or list): Initial momentum which is the
            lower boundary in the cycle for each parameter group.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self,
                 optimizer,
                 base_lr,
                 max_lr,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='auto',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 cycle_momentum=True,
                 momentum_key='momentum',
                 base_momentum=0.8,
                 max_momentum=0.9,
                 last_epoch=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        base_lrs = self._format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

        step_size_up = float(step_size_up)
        step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range', 'auto'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        if mode == 'auto':
            if max_lr / base_lr >= 10:
                mode = 'exp_range'
            else:
                mode = 'triangular'
        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.cycle_momentum = cycle_momentum
        self.momentum_key = momentum_key
        if cycle_momentum:
            if momentum_key not in optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    if momentum_key in group:
                        if momentum_key == 'momentum':
                            group['momentum'] = momentum
                        elif momentum_key == 'betas':
                            betas = group['betas']
                            group['betas'] = betas.__class__([momentum, *betas[1:]])
            if momentum_key == 'momentum':
                self.base_momentums = list(map(lambda group: group['momentum'], optimizer.param_groups))
            elif momentum_key == 'betas':
                self.base_momentums = list(map(lambda group: group['betas'][0], optimizer.param_groups))
            self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)

        super().__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        last_epoch = self.last_epoch
        cycle = math.floor(1 + last_epoch / self.total_size)
        x = 1. + last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                if self.momentum_key == 'momentum':
                    param_group['momentum'] = momentum
                elif self.momentum_key == 'betas':
                    betas = param_group['betas']
                    param_group['betas'] = betas.__class__([momentum, *betas[1:]])

        return lrs


class CyclicStepLR(_LRScheduler):
    r"""
    Class that defines cyclic learning rate that decays the learning rate linearly till the end of cycle and then restarts
    at the maximum value.
    """

    def __init__(self, optimizer, base_lr=0.1, max_lr=0.5, step_size_up=1, step_size_down=4, steps=(
            50, 100, 130, 160, 190, 220, 250, 280), gamma=0.5, step_mode='linear', last_epoch=-1):
        assert len(steps) > 1, 'Please specify step intervals.'
        self.optimizer = optimizer

        base_lrs = self._format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.cycle_len = self.step_size_up + self.step_size_down
        self.steps = steps
        self.gamma = gamma
        assert step_mode in ['linear', 'exp']
        self.step_mode = step_mode
        super().__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self):
        step = len([s for s in self.steps if s <= self.last_epoch])
        gamma = self.gamma ** step
        epoch = self.last_epoch % self.cycle_len

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            min_lr = base_lr * gamma
            max_lr = max_lr * gamma
            if epoch < self.step_size_up:
                if self.step_mode == 'linear':
                    lr = min_lr + epoch / self.step_size_up * (max_lr - min_lr)
                elif self.step_mode == 'exp':
                    lr = min_lr * ((max_lr / min_lr) ** (epoch / self.step_size_up))
            else:
                if self.step_mode == 'linear':
                    lr = min_lr + (self.cycle_len - epoch) / self.step_size_down * (max_lr - min_lr)
                elif self.step_mode == 'exp':
                    lr = min_lr * ((max_lr / min_lr) ** ((self.cycle_len - epoch) / self.step_size_down))
            lrs.append(lr)
        return lrs


class OneCyclicLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        momentum_key (str): momentum for SGD or betas for Adam.
            Default: 'momentum'
        base_momentum (float or list): Initial momentum which is the
            lower boundary in the cycle for each parameter group.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.CyclicLR(optimizer)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self,
                 optimizer,
                 base_lr,
                 max_lr,
                 step_size,
                 mode='linear',
                 end_steps=1,
                 gamma=0.01,
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 last_epoch=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        base_lrs = self._format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

        step_size = float(step_size)
        self.total_size = step_size * 2
        self.step_ratio = 1 / 2

        if mode not in ['linear', 'exp']:
            raise ValueError('mode is invalid.')

        self.mode = mode
        self.end_steps = end_steps
        self.gamma = gamma

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' not in optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    if 'momentum' in group:
                        group['momentum'] = momentum
            self.base_momentums = list(map(lambda group: group['momentum'], optimizer.param_groups))
            self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)

        super().__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        last_epoch = self.last_epoch

        if last_epoch >= self.total_size + self.end_steps:
            lrs = [base_lr * self.gamma for base_lr in self.base_lrs]
        elif last_epoch >= self.total_size:
            epoch = last_epoch - self.total_size
            scale_factor = 1 - epoch / self.end_steps
            lrs = []
            for base_lr in self.base_lrs:
                max_lr = base_lr
                min_lr = base_lr * self.gamma
                if self.mode == 'linear':
                    lr = min_lr + (max_lr - min_lr) * scale_factor
                else:
                    lr = min_lr * ((max_lr / min_lr) ** scale_factor)
                lrs.append(lr)

            if self.cycle_momentum:
                for param_group, max_momentum in zip(self.optimizer.param_groups, self.max_momentums):
                    param_group['momentum'] = max_momentum
        else:
            cycle = math.floor(1 + last_epoch / self.total_size)
            x = 1. + last_epoch / self.total_size - cycle
            if x <= self.step_ratio:
                scale_factor = x / self.step_ratio
            else:
                scale_factor = (x - 1) / (self.step_ratio - 1)

            lrs = []
            for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
                if self.mode == 'linear':
                    lr = base_lr + (max_lr - base_lr) * scale_factor
                else:
                    lr = base_lr * ((max_lr / base_lr) ** scale_factor)
                lrs.append(lr)

            if self.cycle_momentum:
                momentums = []
                for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                    if self.mode == 'linear':
                        momentum = base_momentum + (max_momentum - base_momentum) * scale_factor
                    else:
                        momentum = base_momentum * ((max_momentum / base_momentum) ** scale_factor)
                    momentums.append(momentum)
                for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                    param_group['momentum'] = momentum

        return lrs
