import torch
import pytest
from ai_core.config.model_config import ModelConfig
from ai_core.transformer.model import JarvisTransformer
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.inference.engine import InferenceEngine

def test_kv_cache_numerical_equivalence():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=32
    )
    model = JarvisTransformer(config)
    model.eval()

    prompt_tokens = [10, 15, 20, 25]
    idx_full = torch.tensor([prompt_tokens], dtype=torch.long)

    # Full forward pass
    logits_full, _ = model(idx_full)
    last_logit_full = logits_full[:, -1, :]

    # Incremental KV-Cached pass
    idx_init = torch.tensor([[prompt_tokens[0], prompt_tokens[1], prompt_tokens[2]]], dtype=torch.long)
    logits_init, _, kv_caches = model(idx_init, use_cache=True, start_pos=0)

    next_tok = torch.tensor([[prompt_tokens[3]]], dtype=torch.long)
    logits_step, _, _ = model(next_tok, kv_caches=kv_caches, start_pos=3)
    last_logit_cached = logits_step[:, -1, :]

    # Max difference between logits should be negligible (< 1e-4)
    max_diff = torch.max(torch.abs(last_logit_full - last_logit_cached)).item()
    assert max_diff < 1e-4, f"KV Cache logits mismatch! Max diff: {max_diff}"

def test_inference_engine_kv_cache_speedup():
    config = ModelConfig(
        vocab_size=200,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=64
    )
    model = JarvisTransformer(config)
    tokenizer = BPETokenizer()
    tokenizer.train("JARVIS AI OS system running local models.", vocab_size=200)

    engine = InferenceEngine(model=model, tokenizer=tokenizer, device=torch.device("cpu"))
    generated = engine.generate(prompt="JARVIS", max_new_tokens=10, use_kv_cache=True)
    
    assert isinstance(generated, str)
    assert engine.last_metrics["used_kv_cache"] is True
    assert engine.last_metrics["tokens_generated"] > 0
