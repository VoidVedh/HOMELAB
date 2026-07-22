import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from ai_core.config.model_config import ModelConfig
from ai_core.embeddings.positional import TokenEmbedding, precompute_rope_freqs_cis
from ai_core.layers.norm import RMSNorm
from ai_core.layers.mlp import SwiGLUMLP
from ai_core.attention.multi_head import CausalSelfAttention
from ai_core.transformer.base_model import BaseModel

class TransformerBlock(nn.Module):
    """Transformer decoder block with RMSNorm, Attention (with KV-Cache), and SwiGLU MLP."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv = self.attn(self.attn_norm(x), freqs_cis=freqs_cis, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv

class JarvisTransformer(BaseModel):
    """Native Decoder-only Transformer LLM with KV-Cache support."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.token_emb.embedding.weight = self.lm_head.weight

        # Precompute RoPE Frequencies
        freqs_cis = precompute_rope_freqs_cis(
            dim=config.d_model // config.n_heads,
            end=config.max_seq_len * 4
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_emb.embedding.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
        use_cache: bool = False
    ):
        batch_size, seq_len = idx.shape
        assert start_pos + seq_len <= self.config.max_seq_len, f"Position {start_pos + seq_len} exceeds max_seq_len {self.config.max_seq_len}"

        x = self.token_emb(idx)
        x = self.drop(x)

        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        new_kv_caches = [] if (use_cache or kv_caches is not None) else None

        for i, block in enumerate(self.blocks):
            past_kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, freqs_cis=freqs_cis, kv_cache=past_kv)
            if new_kv_caches is not None:
                new_kv_caches.append(new_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        if new_kv_caches is not None:
            return logits, loss, new_kv_caches
        return logits, loss
