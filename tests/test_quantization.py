import torch
import torch.nn as nn
import pytest

from ai_core.quantization.quant_utils import (
    quantize_fp32_to_nf4,
    dequantize_nf4_to_fp32,
    quantize_fp32_to_int8,
    dequantize_int8_to_fp32
)
from ai_core.quantization.quantized_linear import QuantizedLinear
from ai_core.quantization.qlora import QLoRALinear

def test_nf4_quantization_dequantization_roundtrip():
    weight = torch.randn(64, 128)
    packed, scales = quantize_fp32_to_nf4(weight, block_size=64)
    dequant = dequantize_nf4_to_fp32(packed, scales, weight.shape, block_size=64)

    assert dequant.shape == weight.shape
    mse = torch.mean((weight - dequant) ** 2).item()
    assert mse < 0.05

def test_int8_quantization_dequantization_roundtrip():
    weight = torch.randn(64, 128)
    int8_w, scales = quantize_fp32_to_int8(weight, block_size=64)
    dequant = dequantize_int8_to_fp32(int8_w, scales, block_size=64)

    assert dequant.shape == weight.shape
    mse = torch.mean((weight - dequant) ** 2).item()
    assert mse < 0.05

def test_quantized_linear_forward_and_from_linear():
    orig_linear = nn.Linear(128, 256, bias=True)
    x = torch.randn(4, 128)

    orig_out = orig_linear(x)
    
    q_linear_4bit = QuantizedLinear.from_linear(orig_linear, bits=4)
    q_out_4bit = q_linear_4bit(x)

    assert q_out_4bit.shape == (4, 256)
    
    q_linear_8bit = QuantizedLinear.from_linear(orig_linear, bits=8)
    q_out_8bit = q_linear_8bit(x)

    assert q_out_8bit.shape == (4, 256)

def test_qlora_linear_gradient_propagation():
    orig_linear = nn.Linear(128, 256)
    q_linear = QuantizedLinear.from_linear(orig_linear, bits=4)
    qlora_layer = QLoRALinear(q_linear, r=8, lora_alpha=16.0)

    x = torch.randn(2, 128, requires_grad=True)
    out = qlora_layer(x)
    loss = out.sum()
    loss.backward()

    # Base layer parameters must be frozen (no gradients)
    for p in q_linear.parameters():
        assert p.grad is None

    # LoRA parameters must receive non-zero gradients
    assert qlora_layer.lora_A.grad is not None
    assert qlora_layer.lora_B.grad is not None
    assert qlora_layer.get_trainable_parameters() > 0
    assert qlora_layer.get_frozen_parameters() > 0
