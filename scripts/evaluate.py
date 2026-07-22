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
from ai_core.datasets.multi_domain_corpus import MultiDomainCorpus
from ai_core.evaluator.domain_benchmarks import DomainBenchmarkRunner
from ai_core.evaluator.regression_framework import ModelRegressionFramework
from ai_core.evaluator.benchmark_evaluator import ModelIntelligenceEvaluator
from ai_core.evaluator.task_evaluator import TaskEvaluator
from ai_core.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="HOMELAB Evaluation CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jarvis0/jarvis0_best.pt", help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="config/jarvis0.yaml", help="Path to model config")
    parser.add_argument("--vocab", type=str, default="checkpoints/bpe_vocab.json", help="Path to vocabulary file")
    args = parser.parse_args()

    print("=" * 70)
    print("      HOMELAB RESEARCH EDITION v4.0 EVALUATION SUITE")
    print("=" * 70)

    device = get_device()
    print(f"[*] Target Hardware Device: {device}")

    model_name = os.path.basename(args.config).replace(".yaml", "")
    
    # 1. Load Model & Tokenizer
    if os.path.exists(args.checkpoint):
        model = ModelRegistry.load(model_name, config_dir=os.path.dirname(args.config) or "config", checkpoint_path=args.checkpoint)
        print(f"[*] Reloaded Checkpoint: {args.checkpoint}")
    else:
        model = ModelRegistry.load(model_name, config_dir=os.path.dirname(args.config) or "config")
        print(f"[*] Loaded Base Model: {args.config}")

    if os.path.exists(args.vocab):
        tokenizer = BPETokenizer.load(args.vocab)
        print(f"[*] Loaded BPE Tokenizer: {args.vocab} (Vocab size: {len(tokenizer.vocab)})")
    else:
        tokenizer = BPETokenizer()
        tokenizer.train("JARVIS AI OS multi-domain corpus", vocab_size=model.config.vocab_size)

    # 2. Build 14-Domain Manifests
    print("\n[*] Building 14-Domain Formal Dataset Manifests...")
    manifests = MultiDomainCorpus.generate_all_14_domain_manifests(tokenizer, max_seq_len=model.config.max_seq_len)
    print(f"  [✓] Generated {len(manifests)} formal domain manifests in 'datasets/manifests/':")
    for m in manifests[:5]:
        print(f"      └─ {m['domain']} (Checksum: {m['checksum'][:16]}..., Tokens: {m['tokens']})")
    print(f"      └─ ... and {len(manifests) - 5} more domains.")

    # 3. Execute 7 Benchmark Suites
    print("\n[*] Executing Multi-Category Benchmark Harness (model, coding, reasoning, planning, tool, rag, memory)...")
    runner = DomainBenchmarkRunner(model=model, tokenizer=tokenizer, device=device)
    bench_results = runner.run_suite()

    print(f"  • Overall Success Rate: {bench_results['overall_success_rate_pct']}%")
    print(f"  • Report Log Persisted: {bench_results['report_path']}")

    # 4. Continuous Regression Comparison
    print("\n[*] Running Model Regression Comparison against historical checkpoints...")
    regression = ModelRegressionFramework(history_dir="observability/benchmarks")
    comp = regression.compare_with_previous_run(current_metrics=bench_results)
    
    print(f"  • Regression Check Status : {comp['status']}")
    print(f"  • Regressions Flagged     : {comp['regressions_detected']}")
    if comp["warnings"]:
        for warn in comp["warnings"]:
            print(f"    [!] {warn}")
    else:
        print("  [✓] Zero regressions detected! Performance exceeds or matches prior release.")

    # 5. OS Task Execution Verification
    print("\n[*] Verifying OS Runtime Competence...")
    task_evaluator = TaskEvaluator()
    task_metrics = asyncio.run(task_evaluator.evaluate_task_success())
    print(f"  • Task Success Rate : {task_metrics['success_rate'] * 100}% ({task_metrics['passed_tasks']}/{task_metrics['total_tasks']} tasks passed)")

    print("\n" + "=" * 70)
    print("[✓] v4.0 RESEARCH EDITION EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
