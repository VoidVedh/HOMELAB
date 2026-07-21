import math

class CosineAnnealingWithWarmupScheduler:
    """Cosine learning rate schedule with linear warmup."""
    def __init__(
        self,
        optimizer,
        learning_rate: float,
        min_learning_rate: float,
        warmup_steps: int,
        max_steps: int
    ):
        self.optimizer = optimizer
        self.lr = learning_rate
        self.min_lr = min_learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def step(self, current_step: int):
        if current_step < self.warmup_steps:
            current_lr = self.lr * (current_step + 1) / self.warmup_steps
        elif current_step > self.max_steps:
            current_lr = self.min_lr
        else:
            decay_ratio = (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            current_lr = self.min_lr + coeff * (self.lr - self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        return current_lr
