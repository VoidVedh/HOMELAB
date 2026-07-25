#!/usr/bin/env python3
"""
HOMELAB Milestone v0.8.0 End-to-End Pipeline
Executes Curriculum v3 dataset generation, gradient accumulative training,
experiment tracking, multi-task capability evaluation, and regression gating.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

from ai_core.config.model_config import ModelConfig
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.transformer.model import JarvisTransformer
from ai_core.datasets.curriculum import CurriculumV3Pipeline
from ai_core.datasets.dataset import TextSequenceDataset
from ai_core.trainer.trainer import Trainer
from ai_core.metrics.experiment_tracker import ExperimentTracker
from ai_core.evaluator.instruction_benchmarks import InstructionBenchmarkSuite
from ai_core.evaluator.domain_benchmarks import DomainBenchmarkRunner
from ai_core.evaluator.regression_gate import RegressionGate
from ai_core.utils.device import get_device

def main():
    print("============================================================")
    print("🚀 HOMELAB Research Platform: Milestone v0.8.0 Execution")
    print("============================================================")

    device = get_device()
    print(f"Device: {device}")

    # 1. Dataset Quality & Curriculum v3 Generation
    print("\n--- 1. Generating Curriculum v3 Dataset & Provenance Manifests ---")
    tokenizer = BPETokenizer()
    sample_text = "def add(a, b):\n    return a + b\n" + "<thought>Step 1: Check</thought>\n" * 10
    tokenizer.train(sample_text, vocab_size=1000)

    curriculum = CurriculumV3Pipeline(tokenizer=tokenizer, version="3.0.0")
    curriculum_res = curriculum.build_full_curriculum(output_dir="datasets/manifests/curriculum")
    manifest = curriculum_res["manifest"]
    print(f"Manifest Checksum: {manifest['manifest_checksum'][:16]}...")
    print(f"Processed Samples: {manifest['processed_samples']} (Duplicates Removed: {manifest['duplicates_removed']})")

    # Combine text tokens for dataset training
    combined_text = "\n\n".join([s["text"] for s in curriculum_res["train_split"]])
    tokens = tokenizer.encode(combined_text, add_special_tokens=True)
    train_dataset = TextSequenceDataset(tokens, max_seq_len=2048)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 2. Model Initialization (Jarvis-1 Architecture with GQA & 2048 Context)
    config = ModelConfig(
        name="jarvis1_v080",
        version="0.8.0",
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        d_ff=1024,
        max_seq_len=2048,
        learning_rate=3e-4,
        grad_clip=1.0
    )
    model = JarvisTransformer(config)
    print(f"Initialized Model: {config.name} ({model.get_num_params():,} non-embedding parameters)")

    # 3. Training with Gradient Accumulation (grad_accum_steps=2) & AMP
    print("\n--- 2. Executing Curriculum Training (Gradient Accumulation = 2) ---")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        checkpoint_dir="checkpoints/v080",
        log_dir="training_logs",
        grad_accum_steps=2,
        device=device
    )

    summary = trainer.train_epoch(epoch=1, max_steps=20)
    print(f"Training Complete! Final Loss: {summary['loss']:.4f}, Perplexity: {summary['perplexity']:.4f}")

    ckpt_path = trainer.save_checkpoint(step=20, loss=summary['loss'], filename="jarvis1_v080_step20.pt")
    print(f"Saved Checkpoint to: {ckpt_path}")

    # 4. Multi-Task & Domain Benchmark Evaluation
    print("\n--- 3. Running Multi-Task Capability Evaluation Suite ---")
    instruction_suite = InstructionBenchmarkSuite(model=model, tokenizer=tokenizer, device=device)
    inst_results = instruction_suite.run_suite()
    
    domain_runner = DomainBenchmarkRunner(model=model, tokenizer=tokenizer, device=device)
    domain_results = domain_runner.run_suite()

    print(f"HumanEval-lite Pass@1: {inst_results['humaneval_lite']['pass_at_1']}")
    print(f"MBPP-lite Accuracy: {inst_results['mbpp_lite']['accuracy']}")
    print(f"CoT Reasoning Score: {inst_results['cot_reasoning']['cot_reasoning_score']:.1f}%")
    print(f"Needle-In-Haystack Retrieval: {inst_results['needle_in_haystack']['average_retrieval_pct']:.1f}%")
    print(f"Overall Domain Benchmark Success Rate: {domain_results['overall_success_rate_pct']:.2f}%")

    # 5. Automated Experiment Tracking
    print("\n--- 4. Recording Experiment Metadata & Results ---")
    tracker = ExperimentTracker(experiment_dir="observability/experiments")
    exp_record = tracker.record_experiment(
        model_version="0.8.0",
        dataset_version="3.0.0",
        hyperparameters=config.__dict__,
        training_summary=summary,
        benchmark_results={"instruction": inst_results, "domain": domain_results},
        checkpoint_path=ckpt_path
    )
    print(f"Experiment Logged: {exp_record['report_file']}")

    # 6. Automated Regression Gate
    print("\n--- 5. Evaluating Checkpoint Promotion Gate ---")
    gate = RegressionGate(history_dir="observability/benchmarks")
    gate_decision = gate.evaluate_checkpoint_promotion(
        current_val_loss=summary['loss'],
        current_benchmark_results=domain_results
    )
    print(f"Gate Decision: Status = {gate_decision['status']}, Promoted = {gate_decision['promoted']}")
    print(f"Reason: {gate_decision['reason']}")

    print("\n✅ HOMELAB Milestone v0.8.0 Execution Completed Successfully!")

if __name__ == "__main__":
    main()
