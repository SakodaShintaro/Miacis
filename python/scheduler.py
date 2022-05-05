from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupAndCooldownScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_step, max_step):
        self.warmup_step = warmup_step
        self.max_step = max_step
        super(LinearWarmupAndCooldownScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch <= self.warmup_step:
            # 線形にwarmupする
            return [base_lr * (self.last_epoch / self.warmup_step) for base_lr in self.base_lrs]
        else:
            # 線形にcooldownする
            return [base_lr * (1 - self.last_epoch / self.max_step) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        return super(LinearWarmupAndCooldownScheduler, self).step(epoch)
