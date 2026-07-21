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
from ai_core.evaluator.evaluator import Evaluator
from ai_core.evaluator.task_evaluator import TaskEvaluator
from ai_core.datasets.dataset import TextSequenceDataset
from ai_core.dataloader.dataloader import create_dataloader

def main():
    parser = argparse.ArgumentParser(description="HOMELAB Evaluation CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jarvis0/jarvis0_best.pt", help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="config/jarvis0.yaml", help="Path to model config")
    args = parser.parse_args()

    print("=" * 60)
    print(f"      HOMELAB Evaluation CLI")
    print("=" * 60)

    model_name = os.path.basename(args.config).replace(".yaml", "")
    
    # 1. Load Model
    if os.path.exists(args.checkpoint):
        model = ModelRegistry.load(model_name, config_dir=os.path.dirname(args.config) or "config", checkpoint_path=args.checkpoint)
        print(f"[*] Reloaded Model & Checkpoint: {args.checkpoint}")
    else:
        model = ModelRegistry.load(model_name, config_dir=os.path.dirname(args.config) or "config")
        print(f"[*] Loaded Base Model: {args.config}")

    # 2. Evaluate Validation Dataset Loss & Perplexity
    val_dataset = TextSequenceDataset(list(range(200)), max_seq_len=model.config.max_seq_len)
    val_loader = create_dataloader(val_dataset, batch_size=model.config.batch_size)
    evaluator = Evaluator(model=model, device=torch.device("cpu"))
    val_metrics = evaluator.evaluate(val_loader)

    print(f"[*] Language Model Validation Metrics:")
    print(f"  • Val Loss       : {val_metrics['val_loss']}")
    print(f"  • Val Perplexity : {val_metrics['val_perplexity']}")

    # 3. Evaluate Practical OS Task Execution Success
    print("\n[*] Evaluating Practical OS Task Execution Success Rate...")
    task_evaluator = TaskEvaluator()
    task_metrics = asyncio.run(task_evaluator.evaluate_task_success())
    print(f"  • Success Rate : {task_metrics['success_rate'] * 100}% ({task_metrics['passed_tasks']}/{task_metrics['total_tasks']} tasks passed)")
    for detail in task_metrics["details"]:
        status = "PASSED" if detail["passed"] else "FAILED"
        print(f"    └─ Task '{detail['task_name']}' -> {status}")

    print("\n[✓] Evaluation Complete!")

if __name__ == "__main__":
    main()
