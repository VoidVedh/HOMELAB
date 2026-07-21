import os
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str = "jarvis0"
    version: str = "0.1.0"
    model_type: str = "transformer"
    vocab_size: int = 4096
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    norm_eps: float = 1e-5
    
    # Training Configs
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 1000
    batch_size: int = 16

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found at: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Type conversion helper
        int_fields = {"vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "max_seq_len", "warmup_steps", "max_steps", "batch_size"}
        float_fields = {"dropout", "norm_eps", "learning_rate", "min_learning_rate", "weight_decay", "beta1", "beta2", "grad_clip"}

        cleaned = {}
        for k, v in data.items():
            if k in cls.__dataclass_fields__:
                if k in float_fields and v is not None:
                    cleaned[k] = float(v)
                elif k in int_fields and v is not None:
                    cleaned[k] = int(v)
                else:
                    cleaned[k] = str(v)

        return cls(**cleaned)
