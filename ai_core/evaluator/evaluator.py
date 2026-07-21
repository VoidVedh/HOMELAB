import math
import torch
from typing import Dict, Any
from torch.utils.data import DataLoader
from ai_core.transformer.base_model import BaseModel
from ai_core.utils.device import get_device

class Evaluator:
    """Evaluates BaseModel on validation datasets, computing Loss & Perplexity."""
    def __init__(self, model: BaseModel, device: torch.device = None):
        self.device = device or get_device()
        self.model = model.to(self.device)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, targets=y)
            if loss is not None:
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        perplexity = math.exp(min(avg_loss, 20.0))

        return {
            "val_loss": round(avg_loss, 4),
            "val_perplexity": round(perplexity, 4)
        }
