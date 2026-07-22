import time
import torch
from typing import Optional, List, Dict, Any
from ai_core.transformer.base_model import BaseModel
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.sampler.sampler import LogitSampler
from ai_core.utils.device import Timer, get_device

class InferenceEngine:
    """Autoregressive text generation engine with KV-Cache acceleration and latency profiling."""
    def __init__(self, model: BaseModel, tokenizer: BPETokenizer, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.last_metrics: Dict[str, Any] = {}

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
        use_kv_cache: bool = True
    ) -> str:
        stop_tokens = stop_tokens or [self.tokenizer.special_tokens.get("<eos>", 3)]
        
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not token_ids:
            return ""
        
        idx = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        prompt_len = idx.shape[1]
        
        start_time = time.perf_counter()
        generated_count = 0
        
        if use_kv_cache:
            # 1. Initial Prompt Forward Pass to compute initial KV Caches
            logits, _, kv_caches = self.model(idx, use_cache=True, start_pos=0)
            last_logits = logits[:, -1, :]
            
            next_token = LogitSampler.sample(
                logits=last_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                input_ids=idx
            )
            
            idx = torch.cat((idx, next_token), dim=1)
            generated_count += 1
            
            # 2. Incremental single-token decoding loop using past KV Caches
            while generated_count < max_new_tokens:
                if next_token.item() in stop_tokens:
                    break

                curr_pos = idx.size(1) - 1
                if curr_pos >= self.model.config.max_seq_len:
                    break

                logits, _, kv_caches = self.model(
                    next_token,
                    kv_caches=kv_caches,
                    start_pos=curr_pos
                )
                last_logits = logits[:, -1, :]

                next_token = LogitSampler.sample(
                    logits=last_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    input_ids=idx
                )

                idx = torch.cat((idx, next_token), dim=1)
                generated_count += 1

        else:
            # Full sequence re-computation fallback path
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.model.config.max_seq_len:]
                logits, _ = self.model(idx_cond)
                last_logits = logits[:, -1, :]

                next_token = LogitSampler.sample(
                    logits=last_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    input_ids=idx
                )

                idx = torch.cat((idx, next_token), dim=1)
                generated_count += 1

                if next_token.item() in stop_tokens:
                    break

        elapsed_sec = time.perf_counter() - start_time
        tok_per_sec = generated_count / max(elapsed_sec, 1e-6)
        ms_per_tok = (elapsed_sec / max(generated_count, 1)) * 1000.0
        
        self.last_metrics = {
            "prompt_length": prompt_len,
            "tokens_generated": generated_count,
            "elapsed_sec": round(elapsed_sec, 4),
            "tokens_per_sec": round(tok_per_sec, 2),
            "ms_per_token": round(ms_per_tok, 2),
            "used_kv_cache": use_kv_cache
        }

        generated_tokens = idx[0].cpu().tolist()
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
