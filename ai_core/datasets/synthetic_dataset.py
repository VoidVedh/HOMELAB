import os
from typing import List
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.datasets.dataset import TextSequenceDataset

TECHNICAL_SAMPLES = [
    "def train_model(config, dataloader, device):\n    model = JarvisTransformer(config).to(device)\n    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)\n    return model\n",
    "JARVIS is a fully local AI Operating System running natively on Apple Silicon MPS hardware.\n",
    "The ai_core framework implements decoder-only causal self-attention with RoPE positional embeddings, RMSNorm, and SwiGLU MLP blocks.\n",
    "ExecutivePlanner decomposes goal prompts into DAG execution nodes: step_1_inspect -> step_2_execute -> step_3_log.\n",
    "EventBus publishes TaskSubmittedEvent, TaskCompletedEvent, MemoryStoredEvent across asynchronous worker nodes.\n",
    "TerminalPlugin executes shell commands: pytest -v tests/ --tb=short within sandboxed working directories.\n",
    "FilesystemPlugin supports file operations: read_file, write_file, list_dir, grep_search with security verification.\n",
    "ModelRegistry loads versioned YAML configurations: ModelRegistry.load('jarvis0', config_dir='config').\n",
    "InferenceEngine utilizes KV-Cache past key-value states for single-token autoregressive generation acceleration.\n",
    "LogitSampler applies temperature scaling, top-k, nucleus top-p, and repetition penalties to eliminate text loops.\n"
]

class SyntheticTechnicalCorpus:
    """Synthetic technical dataset generator for code, OS commands, and system architecture training."""

    @staticmethod
    def generate_corpus(multiplier: int = 50) -> str:
        """Generates a rich synthetic corpus string."""
        corpus = []
        for i in range(multiplier):
            idx = i % len(TECHNICAL_SAMPLES)
            corpus.append(f"# [Sample {i+1}]\n{TECHNICAL_SAMPLES[idx]}")
        return "\n".join(corpus)

    @staticmethod
    def create_dataset(
        tokenizer: BPETokenizer,
        max_seq_len: int = 512,
        multiplier: int = 50
    ) -> TextSequenceDataset:
        """Creates a PyTorch TextSequenceDataset from the synthetic technical corpus."""
        raw_text = SyntheticTechnicalCorpus.generate_corpus(multiplier=multiplier)
        token_ids = tokenizer.encode(raw_text, add_special_tokens=True)
        return TextSequenceDataset(token_ids, max_seq_len=max_seq_len)
