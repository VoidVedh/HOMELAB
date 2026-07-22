#!/usr/bin/env python3
"""
CI/CD Automated Model Regression Quality Gate.
Usage: python3 scripts/ci_regression_gate.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_core.registry.model_registry import ModelRegistry
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.evaluator.domain_benchmarks import DomainBenchmarkRunner
from ai_core.evaluator.regression_framework import ModelRegressionFramework
from ai_core.utils.device import get_device

def main():
    print("=" * 70)
    print("      HOMELAB CI/CD AUTOMATED MODEL REGRESSION QUALITY GATE")
    print("=" * 70)

    device = get_device()
    model_file = "checkpoints/jarvis0/jarvis0_best.pt"
    vocab_file = "checkpoints/bpe_vocab.json"

    if not os.path.exists(vocab_file):
        tokenizer = BPETokenizer()
        tokenizer.train("JARVIS AI OS system test corpus", vocab_size=4096)
    else:
        tokenizer = BPETokenizer.load(vocab_file)

    if not os.path.exists(model_file):
        model = ModelRegistry.load("jarvis0", config_dir="config", device=str(device))
    else:
        model = ModelRegistry.load("jarvis0", config_dir="config", checkpoint_path=model_file, device=str(device))

    runner = DomainBenchmarkRunner(model=model, tokenizer=tokenizer, device=device)
    current_run = runner.run_suite()

    regression = ModelRegressionFramework(history_dir="observability/benchmarks", max_regression_pct=5.0)
    comp = regression.compare_with_previous_run(current_metrics=current_run)

    print(f"[*] Overall Success Rate: {current_run['overall_success_rate_pct']}%")
    print(f"[*] Regression Status   : {comp['status']}")

    if comp["regressions_detected"]:
        print("\n[!] CI/CD QUALITY GATE FAILED: Model Regressions Detected!")
        for warn in comp["warnings"]:
            print(f"    └─ {warn}")
        sys.exit(1)
    else:
        print("\n[✓] CI/CD QUALITY GATE PASSED: Zero Regressions Detected.")
        sys.exit(0)

if __name__ == "__main__":
    main()
