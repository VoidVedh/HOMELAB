#!/usr/bin/env python3
"""
HOMELAB Milestone v0.10.0 Benchmark Runner
Measures memory compression ratios, quantization reconstruction errors (MSE),
forward projection latency, and QLoRA adapter gradient propagation.
"""

import time
import torch
import torch.nn as nn

from ai_core.quantization.quant_utils import quantize_fp32_to_nf4, dequantize_nf4_to_fp32
from ai_core.quantization.quantized_linear import QuantizedLinear
from ai_core.quantization.qlora import QLoRALinear

def benchmark_quantization():
    print("============================================================")
    print("⚡ HOMELAB Milestone v0.10.0 Quantization & QLoRA Benchmark")
    print("============================================================")

    # Simulation setup: Large linear layer (4096 in x 4096 out = 16.7M parameters)
    in_features = 4096
    out_features = 4096
    num_params = in_features * out_features

    fp32_weight = torch.randn(out_features, in_features)
    fp32_size_mb = (num_params * 4) / (1024 * 1024)

    # 1. 4-bit NF4 Quantization & Memory Footprint Reduction
    t0 = time.perf_counter()
    packed_nf4, scales = quantize_fp32_to_nf4(fp32_weight, block_size=64)
    quant_time_ms = (time.perf_counter() - t0) * 1000.0

    nf4_size_mb = (packed_nf4.numel() * 1 + scales.numel() * 4) / (1024 * 1024)
    compression_ratio = fp32_size_mb / nf4_size_mb

    dequant_nf4 = dequantize_nf4_to_fp32(packed_nf4, scales, fp32_weight.shape, block_size=64)
    mse_error = torch.mean((fp32_weight - dequant_nf4) ** 2).item()

    print(f"\n--- 1. Memory Compression & Quantization Accuracy ---")
    print(f"Base Parameters: {num_params:,} ({out_features}x{in_features})")
    print(f"FP32 Memory Footprint: {fp32_size_mb:.2f} MB")
    print(f"4-bit NF4 Packed Memory Footprint: {nf4_size_mb:.2f} MB")
    print(f"Memory Compression Ratio: {compression_ratio:.2f}x")
    print(f"Quantization Quantile Error (MSE): {mse_error:.6f}")
    print(f"Quantization Time: {quant_time_ms:.2f} ms")

    # 2. Forward Pass Projection Latency Comparison
    x = torch.randn(16, in_features)

    orig_linear = nn.Linear(in_features, out_features, bias=False)
    orig_linear.weight.data.copy_(fp32_weight)

    q_linear = QuantizedLinear.from_linear(orig_linear, bits=4)

    # Warmup
    _ = orig_linear(x)
    _ = q_linear(x)

    t0 = time.perf_counter()
    for _ in range(50):
        _ = orig_linear(x)
    fp32_latency = ((time.perf_counter() - t0) / 50.0) * 1000.0

    t0 = time.perf_counter()
    for _ in range(50):
        _ = q_linear(x)
    nf4_latency = ((time.perf_counter() - t0) / 50.0) * 1000.0

    print(f"\n--- 2. Forward Projection Latency ---")
    print(f"FP32 Dense Linear Latency: {fp32_latency:.2f} ms")
    print(f"4-bit NF4 Linear Latency: {nf4_latency:.2f} ms")

    # 3. QLoRA Adapter Fine-Tuning Benchmarking
    qlora_layer = QLoRALinear(q_linear, r=16, lora_alpha=32.0)
    trainable_params = qlora_layer.get_trainable_parameters()
    frozen_params = qlora_layer.get_frozen_parameters()
    trainable_ratio = (trainable_params / (trainable_params + frozen_params)) * 100.0

    t0 = time.perf_counter()
    out = qlora_layer(x)
    loss = out.sum()
    loss.backward()
    backward_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n--- 3. QLoRA Fine-Tuning Parameter Efficiency ---")
    print(f"Frozen Base Parameters: {frozen_params:,}")
    print(f"Trainable LoRA Parameters (r=16): {trainable_params:,}")
    print(f"Trainable Parameter Fraction: {trainable_ratio:.4f}%")
    print(f"Forward + Backward Pass Duration: {backward_ms:.2f} ms")
    print(f"Gradient Status: lora_A.grad norm = {qlora_layer.lora_A.grad.norm().item():.4f}, lora_B.grad norm = {qlora_layer.lora_B.grad.norm().item():.4f}")

    print("\n============================================================")
    print("✅ Milestone v0.10.0 Benchmark Completed Successfully!")
    print("============================================================")

def main():
    benchmark_quantization()

if __name__ == "__main__":
    main()
