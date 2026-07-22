import os
import json
import time
import math
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
    """Training manager for BaseModel architectures with structured JSON experiment logging."""
    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "training_logs",
        device: Optional[torch.device] = None
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_dir = log_dir
        
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
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.run_id = f"run_{int(time.time())}"
        self.log_file = os.path.join(self.log_dir, f"{self.run_id}.json")
        self.step_history: list[Dict[str, Any]] = []

    def train_epoch(self, epoch: int, max_steps: Optional[int] = None) -> Dict[str, Any]:
        self.model.train()
        self.metrics.reset()
        step = 0
        start_time = time.perf_counter()

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            _, loss = self.model(x, targets=y)
            loss.backward()

            # Calculate gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip).item()

            self.optimizer.step()
            curr_lr = self.scheduler.step(step)

            num_tokens = x.numel()
            self.metrics.log_step(loss.item(), num_tokens)
            step += 1

            elapsed = time.perf_counter() - start_time
            tokens_per_sec = (self.metrics.total_tokens / max(elapsed, 1e-6))
            
            step_record = {
                "step": step,
                "epoch": epoch,
                "loss": round(loss.item(), 4),
                "perplexity": round(math.exp(min(loss.item(), 20.0)), 4),
                "lr": round(curr_lr, 6),
                "grad_norm": round(grad_norm, 4),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "elapsed_sec": round(elapsed, 4)
            }
            self.step_history.append(step_record)

            if max_steps and step >= max_steps:
                break

        summary = self.metrics.get_summary()
        summary["epoch"] = epoch
        summary["run_id"] = self.run_id
        summary["log_file"] = self.log_file
        
        # Persist structured training log file
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": self.run_id,
                "config": self.config.__dict__,
                "summary": summary,
                "history": self.step_history
            }, f, indent=2)

        return summary

    def save_checkpoint(self, step: int, loss: float, filename: str) -> str:
        ckpt_path = self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            step=step,
            loss=loss,
            filename=filename,
            extra_metadata={"config": self.config.__dict__, "run_id": self.run_id}
        )
        # Update log with latest checkpoint reference
        if self.step_history:
            self.step_history[-1]["checkpoint"] = ckpt_path
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump({
                    "run_id": self.run_id,
                    "config": self.config.__dict__,
                    "latest_checkpoint": ckpt_path,
                    "history": self.step_history
                }, f, indent=2)
        return ckpt_path
