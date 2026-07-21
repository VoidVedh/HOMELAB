import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from ai_core.transformer.base_model import BaseModel
from ai_core.optimizer.optimizer import configure_adamw_optimizer
from ai_core.scheduler.scheduler import CosineAnnealingWithWarmupScheduler
from ai_core.checkpoint.checkpoint_manager import CheckpointManager
from ai_core.metrics.tracker import MetricTracker
from ai_core.utils.device import get_device

class Trainer:
    """Training manager for BaseModel architectures."""
    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints",
        device: Optional[torch.device] = None
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.config = model.config
        self.optimizer = configure_adamw_optimizer(
            model=self.model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        self.scheduler = CosineAnnealingWithWarmupScheduler(
            optimizer=self.optimizer,
            learning_rate=self.config.learning_rate,
            min_learning_rate=self.config.min_learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps
        )
        
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        self.metrics = MetricTracker()

    def train_epoch(self, epoch: int, max_steps: Optional[int] = None) -> Dict[str, Any]:
        self.model.train()
        self.metrics.reset()
        step = 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            _, loss = self.model(x, targets=y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            self.scheduler.step(step)

            num_tokens = x.numel()
            self.metrics.log_step(loss.item(), num_tokens)
            step += 1

            if max_steps and step >= max_steps:
                break

        summary = self.metrics.get_summary()
        summary["epoch"] = epoch
        return summary

    def save_checkpoint(self, step: int, loss: float, filename: str) -> str:
        return self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            step=step,
            loss=loss,
            filename=filename,
            extra_metadata={"config": self.config.__dict__}
        )
