from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any

class BaseModel(nn.Module, ABC):
    """Abstract Base Class for all neural network architectures in ai_core."""

    @abstractmethod
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits and optional loss."""
        pass

    @abstractmethod
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return total parameter count."""
        pass

    def save_checkpoint(self, path: str, extra_metadata: Dict[str, Any] = None):
        """Save model checkpoint with state dict and metadata."""
        payload = {
            "model_state": self.state_dict(),
            "config": getattr(self, "config", None),
            "metadata": extra_metadata or {}
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str):
        """Load model weights from checkpoint file."""
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state"])
        return checkpoint.get("metadata", {})
