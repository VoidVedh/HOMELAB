import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLossWithSmoothing(nn.Module):
    """Cross Entropy Loss supporting label smoothing."""
    def __init__(self, label_smoothing: float = 0.0, ignore_index: int = -1):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index
        )
