#!/usr/bin/env python3
"""
Pre-training Script for Jarvis-0.5 Checkpoint on 14-Domain Technical Corpus.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.registry.model_registry import ModelRegistry
from ai_core.datasets.multi_domain_corpus import MultiDomainCorpus, DOMAINS
from ai_core.dataloader.dataloader import create_dataloader
from ai_core.trainer.trainer import Trainer
from ai_core.inference.engine import InferenceEngine
from ai_core.evaluator.benchmark_evaluator import ModelIntelligenceEvaluator
from ai_core.evaluator.domain_benchmarks import DomainBenchmarkRunner
from ai_core.evaluator.regression_framework import ModelRegressionFramework
from ai_core.utils.device import get_device

def main():
    print("=" * 70)
    print("      JARVIS-0.5 MULTI-DOMAIN PRE-TRAINING RUN")
    print("=" * 70)

    device = get_device()
    print(f"[*] Hardware Acceleration Device: {device}")

    # 1. Load Model & Tokenizer
    model = ModelRegistry.load("jarvis0", config_dir="config", device=str(device))
    num_params = model.get_num_params()
    print(f"[*] Jarvis-0 Model Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    vocab_file = "checkpoints/bpe_vocab.json"
    if os.path.exists(vocab_file):
        tokenizer = BPETokenizer.load(vocab_file)
        print(f"[*] Loaded Vocabulary: {vocab_file} ({len(tokenizer.vocab)} tokens)")
    else:
        tokenizer = BPETokenizer()
        corpus = MultiDomainCorpus.generate_domain_text("Programming", multiplier=50)
        tokenizer.train(corpus, vocab_size=model.config.vocab_size)
        tokenizer.save(vocab_file)
        print(f"[*] Trained Vocabulary ({len(tokenizer.vocab)} tokens)")

    # 2. Build Combined 14-Domain Corpus
    print("[*] Generating Combined 14-Domain Pre-training Corpus...")
    all_chunks = []
    for domain in DOMAINS:
        all_chunks.append(MultiDomainCorpus.generate_domain_text(domain, multiplier=25))
    full_text = "\n\n".join(all_chunks)

    token_ids = tokenizer.encode(full_text, add_special_tokens=True)
    print(f"[*] Total Corpus Tokens: {len(token_ids):,}")

    from ai_core.datasets.dataset import TextSequenceDataset
    train_dataset = TextSequenceDataset(token_ids, max_seq_len=model.config.max_seq_len)
    train_loader = create_dataloader(train_dataset, batch_size=model.config.batch_size, shuffle=True)

    # 3. Instantiate Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        checkpoint_dir="checkpoints/jarvis0",
        log_dir="training_logs",
        device=device
    )

    # 4. Execute Multi-Epoch Pre-training
    print("[*] Starting Pre-Training Loop (8 Epochs)...")
    last_summary = {}
    for epoch in range(1, 9):
        summary = trainer.train_epoch(epoch=epoch, max_steps=40)
        last_summary = summary
        print(f"    Epoch {epoch}/8 | Loss: {summary['loss']:.4f} | Perplexity: {summary['perplexity']:.4f} | Tok/sec: {summary['tokens_per_sec']:.2f}")

    # 5. Save Checkpoints
    v05_path = trainer.save_checkpoint(step=320, loss=last_summary['loss'], filename="jarvis0_v05.pt")
    best_path = trainer.save_checkpoint(step=320, loss=last_summary['loss'], filename="jarvis0_best.pt")
    print(f"[*] Saved Checkpoints:\n    └─ {v05_path}\n    └─ {best_path}")

    # 6. Benchmark & Regression Test
    print("\n[*] Evaluating Model Performance & Regression Check...")
    runner = DomainBenchmarkRunner(model=model, tokenizer=tokenizer, device=device)
    bench_res = runner.run_suite()
    print(f"  • Overall Success Rate: {bench_res['overall_success_rate_pct']}%")

    regression = ModelRegressionFramework(history_dir="observability/benchmarks")
    comp = regression.compare_with_previous_run(current_metrics=bench_res)
    print(f"  • Regression Check    : {comp['status']}")

    print("\n[✓] JARVIS-0.5 Pre-training and Evaluation Successful!")

if __name__ == "__main__":
    main()
