import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ai_core.quantization.quant_utils import (
    quantize_fp32_to_nf4,
    dequantize_nf4_to_fp32,
    quantize_fp32_to_int8,
    dequantize_int8_to_fp32
)

class QuantizedLinear(nn.Module):
    """
    Linear layer storing base weights in 4-bit NormalFloat (NF4) or 8-bit quantized format.
    Provides ~4x to 8x memory footprint reduction while retaining FP32/FP16 forward projection quality.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        block_size: int = 64,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.block_size = block_size
        self.original_shape = torch.Size([out_features, in_features])

        # Initialize temporary FP32 weights for quantization
        weight_temp = torch.randn(out_features, in_features, device=device) / (in_features ** 0.5)

        if bits == 4:
            packed_w, scales = quantize_fp32_to_nf4(weight_temp, block_size=block_size)
            self.register_buffer("packed_weight", packed_w)
            self.register_buffer("scales", scales)
        elif bits == 8:
            int8_w, scales = quantize_fp32_to_int8(weight_temp, block_size=block_size)
            self.register_buffer("packed_weight", int8_w)
            self.register_buffer("scales", scales)
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}. Must be 4 or 8.")

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, bits: int = 4, block_size: int = 64) -> "QuantizedLinear":
        """Factory method converting existing nn.Linear layer to QuantizedLinear."""
        q_layer = cls(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            bias=linear_layer.bias is not None,
            bits=bits,
            block_size=block_size,
            device=linear_layer.weight.device
        )
        
        if bits == 4:
            packed_w, scales = quantize_fp32_to_nf4(linear_layer.weight, block_size=block_size)
        else:
            packed_w, scales = quantize_fp32_to_int8(linear_layer.weight, block_size=block_size)

        q_layer.packed_weight.copy_(packed_w)
        q_layer.scales.copy_(scales)

        if linear_layer.bias is not None and q_layer.bias is not None:
            q_layer.bias.data.copy_(linear_layer.bias.data)

        return q_layer

    def get_dequantized_weight(self) -> torch.Tensor:
        """Dequantizes base weights back to FP32 tensor on the fly."""
        if self.bits == 4:
            return dequantize_nf4_to_fp32(
                packed=self.packed_weight,
                scales=self.scales,
                original_shape=self.original_shape,
                block_size=self.block_size
            )
        else:
            return dequantize_int8_to_fp32(
                int8_weight=self.packed_weight,
                scales=self.scales,
                block_size=self.block_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dequant = self.get_dequantized_weight().to(x.dtype)
        return F.linear(x, w_dequant, self.bias)
