import torch
import torch.nn as nn
from typing import Tuple

def configure_adamw_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95)
) -> torch.optim.AdamW:
    """
    Configure AdamW optimizer applying weight decay only to 2D weight matrices
    (excluding 1D biases and layer norm parameters).
    """
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]

    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
