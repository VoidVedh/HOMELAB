#!/usr/bin/env python3
"""
Executable script to process raw text corpora into cleaned, deduplicated dataset splits & manifests.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.pipeline import DatasetPipeline

def main():
    print("=" * 60)
    print("      JARVIS Dataset Pipeline & Manifest Builder")
    print("=" * 60)

    pipeline = DatasetPipeline(base_dir="datasets")
    
    sample_corpus = """
    JARVIS is an autonomous AI Operating System designed for local Apple Silicon hardware.
    The system decouples a pure machine learning framework ai_core from jarvis_core.
    RAG retrieval pipelines improve accuracy by query rewriting, candidate retrieval, and reranking.
    Context Managers dynamically allocate token budgets across System, Memory, RAG Docs, and State.
    Prompt Compilers compile structured prompts: System, Conversation, RAG Docs, State, User.
    JARVIS maintains a clean research pipeline with versioned reproducible experiment manifests.
    """ * 10

    manifest = pipeline.process_and_create_manifest(
        dataset_name="jarvis_core_corpus",
        raw_text=sample_corpus,
        license_info="MIT / Self-Authored"
    )

    print(f"[*] Dataset Manifest Generated:")
    print(f"  • Name   : {manifest['dataset_name']}")
    print(f"  • Hash   : {manifest['hash'][:16]}...")
    print(f"  • Tokens : {manifest['total_tokens_approx']}")
    print(f"  • Splits : Train={manifest['splits']['train_lines']}, Val={manifest['splits']['val_lines']}, Test={manifest['splits']['test_lines']}")
    print("[✓] Dataset Pipeline Executed Successfully!")

if __name__ == "__main__":
    main()
