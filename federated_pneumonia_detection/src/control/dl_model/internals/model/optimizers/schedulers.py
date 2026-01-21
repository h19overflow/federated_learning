"""
Optimizers and schedulers for deep learning training.

Provides custom learning rate schedulers and optimizer configurations.
"""

import math
from typing import Optional

import torch.optim as optim


class CosineAnnealingWarmupScheduler:
    """
    Cosine Annealing with Linear Warmup scheduler.

    Combines linear warmup with cosine annealing for stable training.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
        warmup_start_lr: float = 1e-7,
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            min_lr: Minimum learning rate after annealing
            warmup_start_lr: Starting learning rate for warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        """Update learning rate based on current epoch."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_epoch < self.warmup_epochs:
                # Linear warmup
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (
                    self.current_epoch / self.warmup_epochs
                )
            else:
                # Cosine annealing
                progress = (self.current_epoch - self.warmup_epochs) / (
                    self.total_epochs - self.warmup_epochs
                )
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
            param_group["lr"] = lr

    def get_last_lr(self):
        """Get current learning rates."""
        return [group["lr"] for group in self.optimizer.param_groups]
