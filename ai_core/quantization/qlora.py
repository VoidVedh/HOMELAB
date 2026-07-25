import math
import torch
import torch.nn as nn
from typing import Optional

from ai_core.quantization.quantized_linear import QuantizedLinear

class QLoRALinear(nn.Module):
    """
    Quantized Low-Rank Adaptation (QLoRA) Linear Layer.
    Combines frozen 4-bit NF4 base weights with 16-bit trainable low-rank adapters (lora_A, lora_B),
    enabling full model fine-tuning with < 1% trainable parameter memory footprint.
    """

    def __init__(
        self,
        base_layer: QuantizedLinear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(r, 1)

        # Freeze base quantized layer parameters
        for p in self.base_layer.parameters():
            p.requires_grad = False

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, base_layer.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

            # Initialize LoRA matrices: A ~ Kaiming Uniform, B = 0
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base quantized linear forward pass (frozen 4-bit / 8-bit projection)
        base_out = self.base_layer(x)

        if self.r > 0:
            # Low-rank adapter branch: Y_adapter = (x @ A.T) @ B.T * scaling
            x_drop = self.dropout(x)
            lora_out = (x_drop @ self.lora_A.T) @ self.lora_B.T
            return base_out + (lora_out * self.scaling)

        return base_out

    def get_trainable_parameters(self) -> int:
        """Returns total trainable parameter count in LoRA adapters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_frozen_parameters(self) -> int:
        """Returns total frozen base weight elements (unpacked FP32 equivalent)."""
        return self.base_layer.in_features * self.base_layer.out_features
