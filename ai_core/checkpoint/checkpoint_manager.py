import os
import torch
from typing import Dict, Any, Optional

class CheckpointManager:
    """Manages model checkpoint serialization, deserialization, and metadata logging."""
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
        loss: float,
        filename: str = "checkpoint.pt",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        filepath = os.path.join(self.checkpoint_dir, filename)
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "step": step,
            "loss": loss,
            "metadata": extra_metadata or {}
        }
        torch.save(state, filepath)
        return filepath

    def load(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if optimizer and checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        return {
            "step": checkpoint.get("step", 0),
            "loss": checkpoint.get("loss", 0.0),
            "metadata": checkpoint.get("metadata", {})
        }
