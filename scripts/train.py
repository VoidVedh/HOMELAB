#!/usr/bin/env python3
"""
Config-driven Training CLI runner for HOMELAB ai_core models.
Usage: python3 scripts/train.py --config config/jarvis0.yaml
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.registry.model_registry import ModelRegistry
from ai_core.datasets.dataset import TextSequenceDataset
from ai_core.dataloader.dataloader import create_dataloader
from ai_core.trainer.trainer import Trainer
from observability.experiment_tracker import ExperimentTracker
from ai_core.utils.device import get_device

def main():
    parser = argparse.ArgumentParser(description="HOMELAB Training CLI")
    parser.add_argument("--config", type=str, default="config/jarvis0.yaml", help="Path to model YAML config")
    args = parser.parse_args()

    print("=" * 60)
    print(f"      HOMELAB Training CLI: {args.config}")
    print("=" * 60)

    device = get_device()
    model_name = os.path.basename(args.config).replace(".yaml", "")
    
    # 1. Load Model via ModelRegistry
    model = ModelRegistry.load(model_name, config_dir=os.path.dirname(args.config) or "config", device=str(device))
    print(f"[*] Loaded Model '{model_name}' ({model.get_num_params() / 1e6:.2f}M params) on {device}")

    # 2. Tokenizer & Dataset Setup
    sample_text = "JARVIS is an autonomous AI Operating System with native PyTorch transformer engine." * 25
    tokenizer = BPETokenizer()
    tokenizer.train(sample_text, vocab_size=model.config.vocab_size)
    
    token_ids = tokenizer.encode(sample_text, add_special_tokens=True)
    dataset = TextSequenceDataset(token_ids, max_seq_len=model.config.max_seq_len)
    loader = create_dataloader(dataset, batch_size=model.config.batch_size, shuffle=True)

    # 3. Train Model
    trainer = Trainer(model=model, train_loader=loader, checkpoint_dir=f"checkpoints/{model_name}", device=device)
    summary = trainer.train_epoch(epoch=1, max_steps=10)
    print(f"[*] Training Summary: Loss = {summary['loss']}, Perplexity = {summary['perplexity']}")

    # 4. Save Checkpoint & Record Experiment
    checkpoint_path = trainer.save_checkpoint(step=10, loss=summary['loss'], filename=f"{model_name}_final.pt")
    
    tracker = ExperimentTracker()
    exp_log = tracker.log_experiment(
        exp_id=f"exp_{model_name}_run",
        config_name=args.config,
        dataset_manifest={"tokens": len(token_ids)},
        checkpoint_path=checkpoint_path,
        metrics=summary,
        notes="Config-driven training pass"
    )

    print(f"[*] Logged Experiment Record: {exp_log}")
    print("[✓] Training Pass Completed Successfully!")

if __name__ == "__main__":
    main()
