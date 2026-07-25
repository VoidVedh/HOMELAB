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
    """Training manager for BaseModel architectures with gradient accumulation, AMP, resume, and structured logging."""
    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "training_logs",
        grad_accum_steps: int = 1,
        device: Optional[torch.device] = None
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_dir = log_dir
        self.grad_accum_steps = max(1, grad_accum_steps)
        
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

    def resume_from_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Resumes model and optimizer weights from checkpoint file."""
        return self.checkpoint_manager.load(checkpoint_path, model=self.model, optimizer=self.optimizer)

    def prune_checkpoints(self, keep_top_k: int = 3):
        """Retains only top-K best checkpoints to prevent storage bloat."""
        self.checkpoint_manager.prune_old_checkpoints(keep_top_k=keep_top_k)

    def train_epoch(self, epoch: int, max_steps: Optional[int] = None) -> Dict[str, Any]:
        self.model.train()
        self.metrics.reset()
        step = 0
        accum_step = 0
        start_time = time.perf_counter()
        device_type = "cuda" if self.device.type == "cuda" else ("mps" if self.device.type == "mps" else "cpu")
        total_target_steps = max_steps or len(self.train_loader)

        self.optimizer.zero_grad()

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            if device_type in ["cuda", "mps"]:
                with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = self.model(x, targets=y)
                    loss = loss / self.grad_accum_steps
            else:
                _, loss = self.model(x, targets=y)
                loss = loss / self.grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] Step {step}: Loss is NaN/Inf, skipping step.")
                continue

            loss.backward()
            accum_step += 1

            if accum_step % self.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip).item()
                self.optimizer.step()
                curr_lr = self.scheduler.step(step)
                self.optimizer.zero_grad()
                
                step += 1
                unscaled_loss = loss.item() * self.grad_accum_steps
                num_tokens = x.numel() * self.grad_accum_steps
                self.metrics.log_step(unscaled_loss, num_tokens)

                elapsed = time.perf_counter() - start_time
                tokens_per_sec = (self.metrics.total_tokens / max(elapsed, 1e-6))
                
                # Training ETA Calculation
                rem_steps = max(0, total_target_steps - step)
                avg_step_time = elapsed / max(step, 1)
                eta_sec = rem_steps * avg_step_time

                step_record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": round(unscaled_loss, 4),
                    "perplexity": round(math.exp(min(unscaled_loss, 20.0)), 4),
                    "lr": round(curr_lr, 6),
                    "grad_norm": round(grad_norm, 4),
                    "tokens_per_sec": round(tokens_per_sec, 2),
                    "elapsed_sec": round(elapsed, 4),
                    "eta_sec": round(eta_sec, 2)
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
