#!/usr/bin/env python3
"""
Speculative Decoding Benchmark Script
Measures generation throughput (tok/sec), latency, acceptance rate, and speedup ratios 
comparing standard autoregressive generation vs speculative decoding.
"""

import time
import torch

from ai_core.config.model_config import ModelConfig
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.transformer.model import JarvisTransformer
from ai_core.inference.engine import InferenceEngine
from ai_core.utils.device import get_device

def main():
    print("============================================================")
    print("⚡ HOMELAB Speculative Decoding & Latency Benchmark")
    print("============================================================")

    device = get_device()
    print(f"Execution Device: {device}")

    # 1. Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(
        "def train_model(config, dataloader, device):\n"
        "    model = JarvisTransformer(config).to(device)\n"
        "    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)\n"
        "    return model\n",
        vocab_size=1000
    )

    # 2. Draft Model (Jarvis-0 small, ~1.5M params)
    draft_config = ModelConfig(
        name="jarvis0_draft",
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=512
    )
    draft_model = JarvisTransformer(draft_config)

    # 3. Target Model (Jarvis-1, ~3.8M params)
    target_config = ModelConfig(
        name="jarvis1_target",
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=2048
    )
    target_model = JarvisTransformer(target_config)

    prompt = "def train_model(config,"
    num_tokens = 32

    # A. Standard Autoregressive Generation
    engine = InferenceEngine(model=target_model, tokenizer=tokenizer, device=device)
    
    t0 = time.perf_counter()
    std_output = engine.generate(prompt=prompt, max_new_tokens=num_tokens, use_kv_cache=True)
    std_elapsed = time.perf_counter() - t0
    std_tok_sec = num_tokens / max(std_elapsed, 1e-6)

    print("\n--- 1. Standard Autoregressive Decoding ---")
    print(f"Output Text: {std_output[:60]}...")
    print(f"Time Elapsed: {std_elapsed:.4f} sec")
    print(f"Throughput: {std_tok_sec:.2f} tokens/sec")

    # B. Speculative Decoding
    t0 = time.perf_counter()
    spec_output = engine.generate(
        prompt=prompt,
        max_new_tokens=num_tokens,
        draft_model=draft_model,
        k_draft=4
    )
    spec_elapsed = time.perf_counter() - t0
    spec_metrics = engine.last_metrics
    spec_tok_sec = spec_metrics["tokens_per_sec"]

    print("\n--- 2. Speculative Decoding (K=4) ---")
    print(f"Output Text: {spec_output[:60]}...")
    print(f"Time Elapsed: {spec_elapsed:.4f} sec")
    print(f"Throughput: {spec_tok_sec:.2f} tokens/sec")
    print(f"Proposed Tokens: {spec_metrics['total_proposed']}")
    print(f"Accepted Tokens: {spec_metrics['total_accepted']}")
    print(f"Acceptance Rate: {spec_metrics['acceptance_rate_pct']:.2f}%")
    print(f"Parallel Target Passes: {spec_metrics['forward_passes']}")

    # Speedup Calculation
    speedup = spec_tok_sec / max(std_tok_sec, 1e-6)
    print("\n============================================================")
    print(f"🚀 Speculative Speedup Ratio: {speedup:.2f}x")
    print("============================================================")

if __name__ == "__main__":
    main()
