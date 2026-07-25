import torch
import pytest
from ai_core.config.model_config import ModelConfig
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.transformer.model import JarvisTransformer
from ai_core.inference.speculative import SpeculativeDecoder
from ai_core.inference.engine import InferenceEngine

def test_speculative_decoding_execution():
    tokenizer = BPETokenizer()
    tokenizer.train("def add(a, b):\n    return a + b\nclass SpeculativeTest:\n    pass\n", vocab_size=500)

    # Draft Model (Jarvis-0 small)
    draft_config = ModelConfig(
        name="jarvis0_draft",
        vocab_size=500,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=512
    )
    draft_model = JarvisTransformer(draft_config)

    # Target Model (Jarvis-1 target)
    target_config = ModelConfig(
        name="jarvis1_target",
        vocab_size=500,
        d_model=256,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=2048
    )
    target_model = JarvisTransformer(target_config)

    decoder = SpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer
    )

    res = decoder.generate_speculative(
        prompt="def add(a, b):",
        max_new_tokens=16,
        k_draft=4,
        temperature=1.0
    )

    assert "text" in res
    assert "tokens" in res
    assert len(res["tokens"]) > 0
    assert res["total_proposed"] > 0
    assert res["acceptance_rate_pct"] >= 0.0
    assert res["tokens_per_sec"] > 0.0

def test_inference_engine_speculative_integration():
    tokenizer = BPETokenizer()
    tokenizer.train("def train():\n    pass\n", vocab_size=500)

    draft_config = ModelConfig(vocab_size=500, d_model=128, n_layers=2, n_heads=4)
    draft_model = JarvisTransformer(draft_config)

    target_config = ModelConfig(vocab_size=500, d_model=256, n_layers=4, n_heads=8, n_kv_heads=2)
    target_model = JarvisTransformer(target_config)

    engine = InferenceEngine(model=target_model, tokenizer=tokenizer)
    output_text = engine.generate(
        prompt="def train():",
        max_new_tokens=12,
        draft_model=draft_model,
        k_draft=3
    )

    assert isinstance(output_text, str)
    assert "tokens_per_sec" in engine.last_metrics
    assert "acceptance_rate_pct" in engine.last_metrics
