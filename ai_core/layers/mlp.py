import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(w1(x)) * w3(x)) w2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
