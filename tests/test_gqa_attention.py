import torch
import pytest
from ai_core.config.model_config import ModelConfig
from ai_core.attention.multi_head import CausalSelfAttention, repeat_kv
from ai_core.transformer.model import JarvisTransformer

def test_gqa_repeat_kv_shape():
    batch, n_kv_heads, seq_len, head_dim = 2, 2, 16, 32
    n_rep = 4
    x = torch.randn(batch, n_kv_heads, seq_len, head_dim)
    repeated = repeat_kv(x, n_rep)
    assert repeated.shape == (batch, n_kv_heads * n_rep, seq_len, head_dim)

def test_gqa_attention_kv_cache_shapes():
    d_model = 256
    n_heads = 8
    n_kv_heads = 2
    seq_len = 10
    head_dim = d_model // n_heads

    attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads)
    x = torch.randn(2, seq_len, d_model)
    output, new_kv = attn(x)
    
    assert output.shape == (2, seq_len, d_model)
    assert new_kv is not None
    k_cached, v_cached = new_kv
    # KV cache should store n_kv_heads, NOT n_heads!
    assert k_cached.shape == (2, n_kv_heads, seq_len, head_dim)
    assert v_cached.shape == (2, n_kv_heads, seq_len, head_dim)

def test_gqa_mha_equivalence_when_equal_heads():
    d_model = 128
    n_heads = 4
    n_kv_heads = 4
    attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads)
    x = torch.randn(1, 5, d_model)
    out, kv = attn(x)
    assert out.shape == (1, 5, d_model)

def test_jarvis_transformer_gqa_forward_and_backward():
    config = ModelConfig(
        name="jarvis1_test",
        d_model=256,
        n_heads=8,
        n_kv_heads=2,
        n_layers=2,
        max_seq_len=2048,
        vocab_size=1000
    )
    model = JarvisTransformer(config)
    idx = torch.randint(0, 1000, (2, 64))
    targets = torch.randint(0, 1000, (2, 64))

    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, 64, 1000)
    assert loss is not None
    loss.backward()
    
    # Verify gradient flow to q_proj and k_proj
    assert model.blocks[0].attn.q_proj.weight.grad is not None
    assert model.blocks[0].attn.k_proj.weight.grad is not None
