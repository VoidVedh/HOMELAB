import torch
import pytest
from ai_core.config.model_config import ModelConfig
from ai_core.transformer.model import JarvisTransformer
from ai_core.registry.model_registry import ModelRegistry

def test_jarvis_transformer_shapes():
    config = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=64
    )
    model = JarvisTransformer(config)
    assert model.get_num_params() > 0
    
    batch_size = 2
    seq_len = 16
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    assert loss.item() > 0.0

def test_model_registry_load():
    model = ModelRegistry.load("jarvis0", config_dir="config", device="cpu")
    assert isinstance(model, JarvisTransformer)
    assert model.config.name == "jarvis0"
