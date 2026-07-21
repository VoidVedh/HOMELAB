#!/usr/bin/env python3
"""
Train Jarvis-0 mini proof-of-concept model (~5.2M parameters) using ai_core framework.
"""
import os
import sys
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.registry.model_registry import ModelRegistry
from ai_core.datasets.dataset import TextSequenceDataset
from ai_core.dataloader.dataloader import create_dataloader
from ai_core.trainer.trainer import Trainer
from ai_core.inference.engine import InferenceEngine
from ai_core.utils.device import get_device

def main():
    print("=" * 60)
    print("      JARVIS-0 Baseline Model Training & Validation")
    print("=" * 60)

    device = get_device()
    print(f"[*] Target Device: {device}")

    # 1. Prepare Training Corpus
    sample_text = """
    JARVIS is a fully local AI Operating System running on Apple Silicon.
    JARVIS uses an executive planner to decompose complex user prompts into structured tasks.
    JARVIS contains an event bus that connects agents, tools, terminal runners, and memory modules.
    The ai_core framework implements a custom decoder-only transformer architecture with Causal Self-Attention,
    RMSNorm, SwiGLU Feed-Forward Networks, and Rotary Positional Embeddings (RoPE).
    JARVIS learns from past execution experiences, terminal command results, and code edits.
    The primary objective of JARVIS is to create an open-source, reproducible AI research system.
    """ * 20  # Repeat to create sufficient corpus for mini model

    # 2. Train Tokenizer
    print("[*] Training BPE Tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.train(sample_text, vocab_size=4096)
    tokenizer.save("checkpoints/bpe_vocab.json")
    print(f"[*] Tokenizer Vocab Size: {len(tokenizer.vocab)}")

    # 3. Instantiate Jarvis-0 Model from config/jarvis0.yaml
    print("[*] Instantiating Jarvis-0 via ModelRegistry...")
    model = ModelRegistry.load("jarvis0", config_dir="config", device=str(device))
    num_params = model.get_num_params()
    print(f"[*] Jarvis-0 Model Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # 4. Prepare Dataset & DataLoader
    token_ids = tokenizer.encode(sample_text, add_special_tokens=True)
    dataset = TextSequenceDataset(token_ids, max_seq_len=model.config.max_seq_len)
    loader = create_dataloader(dataset, batch_size=model.config.batch_size, shuffle=True)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=loader,
        checkpoint_dir="checkpoints/jarvis0",
        device=device
    )

    # 6. Train for 5 epochs
    print("[*] Starting Training Loop...")
    for epoch in range(1, 6):
        summary = trainer.train_epoch(epoch=epoch, max_steps=20)
        print(f"    Epoch {epoch}/5 | Loss: {summary['loss']} | Perplexity: {summary['perplexity']} | Tok/sec: {summary['tokens_per_sec']}")

    # 7. Save Model Checkpoint
    checkpoint_path = trainer.save_checkpoint(step=100, loss=summary['loss'], filename="jarvis0_best.pt")
    print(f"[*] Saved Checkpoint: {checkpoint_path}")

    # 8. Test Native Inference Engine
    print("\n[*] Testing Autoregressive Text Generation via InferenceEngine...")
    engine = InferenceEngine(model=model, tokenizer=tokenizer, device=device)
    prompt = "JARVIS is"
    generated = engine.generate(prompt=prompt, max_new_tokens=32, temperature=0.7)
    print("-" * 60)
    print(f"Prompt   : {prompt}")
    print(f"Generated: {generated}")
    print("-" * 60)
    print("[✓] JARVIS-0 Training & Native Model Generation Successful!")

if __name__ == "__main__":
    main()
