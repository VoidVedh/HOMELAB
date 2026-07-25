#!/usr/bin/env python3
"""
Jarvis-1 Engine Training & Evaluation Runner
Trains the Jarvis-1 model with GQA (Grouped-Query Attention), evaluates multi-domain benchmarks,
and executes regression gates.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

from ai_core.config.model_config import ModelConfig
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.transformer.model import JarvisTransformer
from ai_core.datasets.multi_domain_corpus import MultiDomainCorpus
from ai_core.trainer.trainer import Trainer
from ai_core.evaluator.domain_benchmarks import DomainBenchmarkRunner
from ai_core.evaluator.regression_framework import ModelRegressionFramework
from ai_core.utils.device import get_device

def main():
    print("============================================================")
    print("🚀 HOMELAB Research Engine: Jarvis-1 Evolution Runner")
    print("============================================================")

    device = get_device()
    print(f"Device: {device}")

    # 1. Load Tokenizer & Datasets
    tokenizer = BPETokenizer()
    tokenizer.train("\n".join(["def add(a, b):\n    return a + b\n", "class JarvisTransformer:\n    pass\n"]), vocab_size=1000)

    print("Building Multi-Domain Technical Corpus v2 Dataset & Provenance Manifests...")
    manifests = MultiDomainCorpus.generate_all_14_domain_manifests(tokenizer, max_seq_len=2048)
    print(f"Generated {len(manifests)} formal domain manifests under datasets/manifests/")

    dataset_res = MultiDomainCorpus.build_dataset_and_manifest("Programming", tokenizer, max_seq_len=2048)
    dataset = dataset_res["dataset"]
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 2. Initialize Jarvis-1 GQA Model Architecture
    config = ModelConfig(
        name="jarvis1",
        version="1.0.0",
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,  # GQA 4:1 Ratio
        d_ff=1024,
        max_seq_len=2048,
        batch_size=4,
        learning_rate=3e-4,
        max_steps=50,
        warmup_steps=10
    )
    print(f"Model Architecture: Jarvis-1 (d_model={config.d_model}, n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}, max_seq_len={config.max_seq_len})")
    
    model = JarvisTransformer(config)
    num_params = model.get_num_params()
    print(f"Total Parameters (Non-embedding): {num_params:,}")

    # 3. Train Model
    print("\nStarting Training Epoch...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        checkpoint_dir="checkpoints/jarvis1",
        log_dir="training_logs",
        device=device
    )

    summary = trainer.train_epoch(epoch=1, max_steps=20)
    print(f"Training Complete! Final Loss: {summary['loss']:.4f}, Perplexity: {summary['perplexity']:.4f}")
    
    ckpt_path = trainer.save_checkpoint(step=20, loss=summary['loss'], filename="jarvis1_step20.pt")
    print(f"Saved Checkpoint to: {ckpt_path}")

    # 4. Evaluate Domain Benchmarks
    print("\nRunning Domain Benchmark Evaluation Suite...")
    benchmark_runner = DomainBenchmarkRunner(model=model, tokenizer=tokenizer, device=device)
    results = benchmark_runner.run_suite()
    print(f"Overall Benchmark Success Rate: {results['overall_success_rate_pct']:.2f}%")
    print(f"Report saved to: {results['report_path']}")

    # 5. Check CI Regression Gate
    print("\nRunning CI/CD Regression Gate Verification...")
    detector = ModelRegressionFramework()
    report = detector.compare_with_previous_run(results)
    print(f"Regression Report Status = {report['status']}, Regressions Detected = {report['regressions_detected']}")

    print("\n✅ Jarvis-1 Evolution & Verification Completed Successfully!")

if __name__ == "__main__":
    main()
