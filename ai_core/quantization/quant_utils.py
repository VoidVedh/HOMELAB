import torch
from typing import Tuple

# 4-bit NormalFloat (NF4) quantile lookup values for standard normal distribution N(0, 1)
NF4_VALUES = torch.tensor([
    -1.0,
    -0.6961928009986877,
    -0.5251643657684326,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.2471167892217636,
    0.3396162688732147,
    0.44071598649024963,
    0.5581144094467163,
    0.6992796063423157,
    1.0
], dtype=torch.float32)

def quantize_fp32_to_nf4(weight: torch.Tensor, block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes FP32 weight tensor into 4-bit NormalFloat (NF4) packed uint8 tensor and block scales.
    Returns: (packed_uint8_weights, block_scales)
    """
    shape = weight.shape
    flat_w = weight.detach().flatten()
    numel = flat_w.numel()

    # Pad flat_w if not multiple of block_size
    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        flat_w = torch.cat([flat_w, torch.zeros(pad_len, device=weight.device, dtype=weight.dtype)])

    num_blocks = flat_w.numel() // block_size
    blocks = flat_w.view(num_blocks, block_size)

    # Compute block scale: s = max(|W_block|)
    scales = torch.abs(blocks).max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    norm_blocks = blocks / scales

    # Map normalized block values to nearest NF4 index (0..15)
    nf4_lut = NF4_VALUES.to(weight.device)
    diffs = torch.abs(norm_blocks.unsqueeze(-1) - nf4_lut) # (num_blocks, block_size, 16)
    indices = torch.argmin(diffs, dim=-1).to(torch.uint8) # (num_blocks, block_size)

    flat_indices = indices.view(-1)
    
    # Pack two 4-bit indices per uint8 byte
    # Even indices occupy low 4 bits, odd indices occupy high 4 bits
    if flat_indices.numel() % 2 != 0:
        flat_indices = torch.cat([flat_indices, torch.zeros(1, dtype=torch.uint8, device=weight.device)])

    low_bits = flat_indices[0::2]
    high_bits = flat_indices[1::2] << 4
    packed = low_bits | high_bits

    return packed, scales.squeeze(-1)

def dequantize_nf4_to_fp32(
    packed: torch.Tensor,
    scales: torch.Tensor,
    original_shape: torch.Size,
    block_size: int = 64
) -> torch.Tensor:
    """
    Dequantizes 4-bit NF4 packed uint8 tensor and block scales back to FP32 tensor.
    """
    device = packed.device
    nf4_lut = NF4_VALUES.to(device)

    # Unpack uint8 byte into low and high 4-bit indices
    low_bits = packed & 0x0F
    high_bits = (packed >> 4) & 0x0F

    unpacked_indices = torch.stack([low_bits, high_bits], dim=-1).flatten()
    total_elements = scales.numel() * block_size
    unpacked_indices = unpacked_indices[:total_elements]

    blocks_indices = unpacked_indices.view(scales.numel(), block_size).long()
    dequantized_blocks = nf4_lut[blocks_indices] * scales.unsqueeze(-1)

    flat_dequantized = dequantized_blocks.flatten()
    original_numel = original_shape.numel()

    return flat_dequantized[:original_numel].view(original_shape)

def quantize_fp32_to_int8(weight: torch.Tensor, block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes FP32 weight tensor into signed 8-bit integer tensor and block scales.
    Returns: (int8_weights, block_scales)
    """
    flat_w = weight.detach().flatten()
    numel = flat_w.numel()

    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        flat_w = torch.cat([flat_w, torch.zeros(pad_len, device=weight.device, dtype=weight.dtype)])

    num_blocks = flat_w.numel() // block_size
    blocks = flat_w.view(num_blocks, block_size)

    scales = torch.abs(blocks).max(dim=-1, keepdim=True).values.clamp(min=1e-8) / 127.0
    int8_blocks = torch.round(blocks / scales).clamp(-128, 127).to(torch.int8)

    return int8_blocks.view(-1)[:numel].view(weight.shape), scales.squeeze(-1)

def dequantize_int8_to_fp32(
    int8_weight: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 64
) -> torch.Tensor:
    """
    Dequantizes 8-bit integer tensor and block scales back to FP32 tensor.
    """
    flat_w = int8_weight.detach().flatten().to(torch.float32)
    num_blocks = scales.numel()
    
    pad_len = (num_blocks * block_size) - flat_w.numel()
    if pad_len > 0:
        flat_w = torch.cat([flat_w, torch.zeros(pad_len, device=int8_weight.device)])

    blocks = flat_w.view(num_blocks, block_size)
    dequantized = blocks * scales.unsqueeze(-1)

    return dequantized.flatten()[:int8_weight.numel()].view(int8_weight.shape)
