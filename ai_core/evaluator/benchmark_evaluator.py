import math
import time
import torch
from typing import Dict, Any, Optional

from ai_core.transformer.base_model import BaseModel
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.inference.engine import InferenceEngine
from ai_core.datasets.dataset import TextSequenceDataset
from ai_core.dataloader.dataloader import create_dataloader

class ModelIntelligenceEvaluator:
    """Standardized Model Intelligence & Inference Evaluator for HOMELAB ai_core."""

    def __init__(self, model: BaseModel, tokenizer: BPETokenizer, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    @torch.no_grad()
    def evaluate_loss_and_perplexity(self, text_dataset: TextSequenceDataset, batch_size: int = 16) -> Dict[str, float]:
        """Measures validation loss and perplexity across dataset."""
        self.model.eval()
        loader = create_dataloader(text_dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        total_batches = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, targets=y)
            if loss is not None:
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        perplexity = math.exp(min(avg_loss, 20.0))
        return {
            "validation_loss": round(avg_loss, 4),
            "perplexity": round(perplexity, 4)
        }

    @torch.no_grad()
    def benchmark_kv_cache_speedup(
        self,
        prompt: str = "JARVIS is an AI Operating System",
        max_new_tokens: int = 32
    ) -> Dict[str, Any]:
        """Measures empirical generation throughput and speedup multiplier comparing KV-Cache vs baseline."""
        engine = InferenceEngine(model=self.model, tokenizer=self.tokenizer, device=self.device)

        # Baseline generation (Without KV-Cache)
        _ = engine.generate(prompt=prompt, max_new_tokens=max_new_tokens, use_kv_cache=False)
        baseline_metrics = dict(engine.last_metrics)

        # Accelerated generation (With KV-Cache)
        _ = engine.generate(prompt=prompt, max_new_tokens=max_new_tokens, use_kv_cache=True)
        cached_metrics = dict(engine.last_metrics)

        speedup_ratio = cached_metrics["tokens_per_sec"] / max(baseline_metrics["tokens_per_sec"], 1e-6)

        return {
            "baseline_tok_sec": baseline_metrics["tokens_per_sec"],
            "kv_cached_tok_sec": cached_metrics["tokens_per_sec"],
            "baseline_ms_tok": baseline_metrics.get("avg_latency_per_tok_ms", baseline_metrics.get("ms_per_token", 0.0)),
            "kv_cached_ms_tok": cached_metrics.get("avg_latency_per_tok_ms", cached_metrics.get("ms_per_token", 0.0)),
            "speedup_multiplier": round(speedup_ratio, 2)
        }

    def evaluate_tokenizer_fidelity(self, sample_texts: list[str]) -> Dict[str, Any]:
        """Measures roundtrip fidelity and unknown token frequency across test sentences."""
        total_tokens = 0
        unk_tokens = 0
        exact_matches = 0

        for text in sample_texts:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            decoded = self.tokenizer.decode(encoded, skip_special_tokens=True)
            
            total_tokens += len(encoded)
            unk_tokens += encoded.count(self.tokenizer.special_tokens["<unk>"])
            
            if text.strip() == decoded.strip():
                exact_matches += 1

        unk_rate_pct = (unk_tokens / max(total_tokens, 1)) * 100.0
        fidelity_pct = (exact_matches / max(len(sample_texts), 1)) * 100.0

        return {
            "total_tokens_evaluated": total_tokens,
            "unk_token_count": unk_tokens,
            "unk_rate_pct": round(unk_rate_pct, 4),
            "roundtrip_fidelity_pct": round(fidelity_pct, 2)
        }
