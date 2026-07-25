import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ai_core.embeddings.positional import apply_rotary_emb

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat Key and Value heads n_rep times to match Query head count in GQA."""
    if n_rep == 1:
        return x
    batch_size, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch_size, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch_size, n_kv_heads * n_rep, seq_len, head_dim)
    )

class CausalSelfAttention(nn.Module):
    """Causal Multi-Head / Grouped-Query Self-Attention (GQA) with RoPE and KV-Cache support."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.n_rep = n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Projections: (batch, seq_len, n_heads/n_kv_heads, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # Transpose for attention computation: (batch, n_heads/n_kv_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle KV Cache concatenation for incremental autoregressive generation
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v)
        total_len = k.size(2)

        # Repeat key/value heads if using GQA
        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        # Fast Scaled Dot-Product Attention
        if seq_len == 1 and kv_cache is not None:
            # Single token decoding step using cached K/V: no causal mask required
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            context = torch.matmul(attn_weights, v_rep)
        else:
            # Full sequence forward pass (training / prompt encoding)
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if seq_len > 1:
                causal_mask = torch.triu(
                    torch.ones(seq_len, total_len, device=x.device), diagonal=1 + (total_len - seq_len)
                ).bool()
                scores = scores.masked_fill(causal_mask, float("-inf"))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            context = torch.matmul(attn_weights, v_rep)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.resid_dropout(self.out_proj(context))
        return output, new_kv_cache
