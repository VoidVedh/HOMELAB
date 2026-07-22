import math
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Set

from ai_core.transformer.model import JarvisTransformer

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) linear layer wrapper.
    Decomposes weight updates into rank r matrices: W_new = W_base + (B @ A) * (alpha / r).
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(r, 1)

        # Freeze base linear layer weights
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.lora_A = None
            self.lora_B = None

        self.merged = False

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merges LoRA low-rank weights directly into base weight matrix for zero-latency inference."""
        if not self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_linear.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Unmerges LoRA low-rank weights from base weight matrix."""
        if self.merged and self.r > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_linear.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or self.r == 0:
            return self.base_linear(x)
        
        base_out = self.base_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling

class LoRAModel(nn.Module):
    """
    Parameter-Efficient Fine-Tuning (PEFT) wrapper for JarvisTransformer.
    Injects LoRA adapters into designated target modules (default: q_proj, v_proj).
    """
    def __init__(
        self,
        model: JarvisTransformer,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    ):
        super().__init__()
        self.model = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules

        # Freeze all parameters in original model
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace target linear projections with LoRALinear wrappers
        self.lora_layers: Dict[str, LoRALinear] = {}
        self._inject_lora_layers(self.model, r, lora_alpha, lora_dropout, target_modules)

    def _inject_lora_layers(
        self,
        module: nn.Module,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
        target_modules: Tuple[str, ...]
    ):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(t in name for t in target_modules):
                lora_layer = LoRALinear(
                    base_linear=child,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout
                )
                setattr(module, name, lora_layer)
                self.lora_layers[name] = lora_layer
            else:
                self._inject_lora_layers(child, r, lora_alpha, lora_dropout, target_modules)

    def merge_adapter(self):
        """Merges all LoRA adapters for zero-overhead inference."""
        for layer in self.lora_layers.values():
            layer.merge()

    def unmerge_adapter(self):
        """Unmerges all LoRA adapters."""
        for layer in self.lora_layers.values():
            layer.unmerge()

    def get_trainable_params(self) -> Tuple[int, int, float]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_pct = (trainable_params / max(total_params, 1)) * 100.0
        return total_params, trainable_params, round(trainable_pct, 4)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save_lora_adapter(self, file_path: str):
        """Saves only the lightweight LoRA adapter state dictionary and configuration."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        adapter_state = {k: v.cpu() for k, v in self.state_dict().items() if "lora_" in k}
        config = {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules
        }
        torch.save({"config": config, "state_dict": adapter_state}, file_path)

    @classmethod
    def load_lora_adapter(cls, model: JarvisTransformer, file_path: str) -> "LoRAModel":
        """Loads a saved LoRA adapter onto a base model instance."""
        checkpoint = torch.load(file_path, map_location="cpu")
        config = checkpoint["config"]
        lora_model = cls(
            model=model,
            r=config["r"],
            lora_alpha=config["lora_alpha"],
            target_modules=tuple(config["target_modules"])
        )
        lora_model.load_state_dict(checkpoint["state_dict"], strict=False)
        return lora_model
