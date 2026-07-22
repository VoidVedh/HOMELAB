#!/usr/bin/env python3
"""
Standalone Evaluation & Benchmark CLI for HOMELAB ai_core models and JARVIS OS runtime tasks.
Usage: python3 scripts/evaluate.py --checkpoint checkpoints/jarvis0/jarvis0_best.pt --config config/jarvis0.yaml
"""
import os
import sys
import argparse
import asyncio
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_core.registry.model_registry import ModelRegistry
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.evaluator.evaluator import Evaluator
from ai_core.evaluator.task_evaluator import TaskEvaluator
from ai_core.evaluator.benchmark_evaluator import ModelIntelligenceEvaluator
from ai_core.datasets.synthetic_dataset import SyntheticTechnicalCorpus
from ai_core.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="HOMELAB Evaluation CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jarvis0/jarvis0_best.pt", help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="config/jarvis0.yaml", help="Path to model config")
    parser.add_argument("--vocab", type=str, default="checkpoints/bpe_vocab.json", help="Path to vocabulary file")
    args = parser.parse_args()

    print("=" * 65)
    print("      HOMELAB MODEL INTELLIGENCE & INFERENCE EVALUATION CLI")
    print("=" * 65)

    device = get_device()
    print(f"[*] Target Hardware Device: {device}")

    model_name = os.path.basename(args.config).replace(".yaml", "")
    
    # 1. Load Model & Tokenizer
    if os.path.exists(args.checkpoint):
        model = ModelRegistry.load(model_name, config_dir=os.path.dirname(args.config) or "config", checkpoint_path=args.checkpoint)
        print(f"[*] Reloaded Model & Checkpoint: {args.checkpoint}")
    else:
        model = ModelRegistry.load(model_name, config_dir=os.path.dirname(args.config) or "config")
        print(f"[*] Loaded Base Model: {args.config}")

    if os.path.exists(args.vocab):
        tokenizer = BPETokenizer.load(args.vocab)
        print(f"[*] Loaded BPE Tokenizer: {args.vocab} (Vocab size: {len(tokenizer.vocab)})")
    else:
        tokenizer = BPETokenizer()
        tokenizer.train(SyntheticTechnicalCorpus.generate_corpus(multiplier=5), vocab_size=model.config.vocab_size)
        print(f"[*] Initialized & Trained Synthetic Tokenizer (Vocab size: {len(tokenizer.vocab)})")

    # 2. Evaluate Synthetic Technical Corpus Loss & Perplexity
    val_dataset = SyntheticTechnicalCorpus.create_dataset(tokenizer, max_seq_len=model.config.max_seq_len, multiplier=10)
    intel_evaluator = ModelIntelligenceEvaluator(model=model, tokenizer=tokenizer, device=device)
    loss_metrics = intel_evaluator.evaluate_loss_and_perplexity(val_dataset, batch_size=model.config.batch_size)

    print("\n[*] Model Validation Metrics:")
    print(f"  • Validation Loss : {loss_metrics['validation_loss']}")
    print(f"  • Perplexity      : {loss_metrics['perplexity']}")

    # 3. Tokenizer Fidelity & Unknown Token Rate
    sample_texts = [
        "JARVIS is a fully local AI Operating System running on Apple Silicon.",
        "def train_model(config, dataloader): return JarvisTransformer(config)",
        "TerminalPlugin.execute('pytest -v tests/')"
    ]
    fid_metrics = intel_evaluator.evaluate_tokenizer_fidelity(sample_texts)
    print("\n[*] Tokenizer Fidelity & Lossless Benchmark:")
    print(f"  • Evaluated Tokens: {fid_metrics['total_tokens_evaluated']}")
    print(f"  • Unknown Rate    : {fid_metrics['unk_rate_pct']}% (<unk> count: {fid_metrics['unk_token_count']})")
    print(f"  • Fidelity Match  : {fid_metrics['roundtrip_fidelity_pct']}%")

    # 4. KV-Cache Speedup Benchmark
    speedup_metrics = intel_evaluator.benchmark_kv_cache_speedup(prompt="JARVIS is", max_new_tokens=32)
    print("\n[*] Inference Engine & KV-Cache Speedup Benchmark:")
    print(f"  • Baseline Throughput : {speedup_metrics['baseline_tok_sec']} tok/sec ({speedup_metrics['baseline_ms_tok']} ms/tok)")
    print(f"  • KV-Cached Throughput: {speedup_metrics['kv_cached_tok_sec']} tok/sec ({speedup_metrics['kv_cached_ms_tok']} ms/tok)")
    print(f"  • Speedup Multiplier  : {speedup_metrics['speedup_multiplier']}x")

    # 5. OS Task Execution Success Rate
    print("\n[*] Evaluating Practical OS Task Execution Success Rate...")
    task_evaluator = TaskEvaluator()
    task_metrics = asyncio.run(task_evaluator.evaluate_task_success())
    print(f"  • Task Success Rate : {task_metrics['success_rate'] * 100}% ({task_metrics['passed_tasks']}/{task_metrics['total_tasks']} tasks passed)")
    for detail in task_metrics["details"]:
        status = "PASSED" if detail["passed"] else "FAILED"
        print(f"    └─ Task '{detail['task_name']}' -> {status}")

    print("\n[✓] Evaluation Completed Successfully!")

if __name__ == "__main__":
    main()
