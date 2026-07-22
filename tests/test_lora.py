import os
import torch
import pytest
from ai_core.config.model_config import ModelConfig
from ai_core.transformer.model import JarvisTransformer
from ai_core.lora.lora import LoRAModel, LoRALinear

def test_lora_parameter_freezing_and_injection():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=32
    )
    base_model = JarvisTransformer(config)
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params_before = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    assert trainable_params_before == total_params

    # Wrap model with LoRA (rank r=4)
    lora_model = LoRAModel(base_model, r=4, lora_alpha=8.0, target_modules=("q_proj", "v_proj"))
    t_tot, t_train, t_pct = lora_model.get_trainable_params()

    assert t_train < t_tot
    assert t_pct < 10.0  # Less than 10% trainable parameters
    assert t_train > 0

def test_lora_forward_pass_and_merge():
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=32
    )
    base_model = JarvisTransformer(config)
    base_model.eval()
    
    lora_model = LoRAModel(base_model, r=4, lora_alpha=8.0)
    
    idx = torch.randint(0, config.vocab_size, (2, 16))
    
    # Forward pass before merging
    logits_unmerged, _ = lora_model(idx)
    
    # Merge adapter into base weights
    lora_model.merge_adapter()
    logits_merged, _ = lora_model(idx)
    
    # Merged vs unmerged outputs should match within float precision
    max_diff = torch.max(torch.abs(logits_unmerged - logits_merged)).item()
    assert max_diff < 1e-4

def test_lora_save_and_load_adapter(tmp_path):
    config = ModelConfig(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=32
    )
    base_model = JarvisTransformer(config)
    lora_model = LoRAModel(base_model, r=4, lora_alpha=8.0)

    save_path = str(tmp_path / "lora_adapter.pt")
    lora_model.save_lora_adapter(save_path)
    assert os.path.exists(save_path)

    # Reload adapter on fresh base model
    fresh_base = JarvisTransformer(config)
    reloaded_lora = LoRAModel.load_lora_adapter(fresh_base, save_path)
    assert reloaded_lora.r == 4
    assert reloaded_lora.lora_alpha == 8.0
