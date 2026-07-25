import time
import torch
from typing import Optional, List, Dict, Any, Generator, Callable
from ai_core.transformer.base_model import BaseModel
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.sampler.sampler import LogitSampler
from ai_core.inference.streamer import BaseStreamer, TokenStreamer
from ai_core.utils.device import get_device

class InferenceEngine:
    """Autoregressive text generation engine with streaming KV-Cache acceleration and latency profiling."""
    def __init__(self, model: BaseModel, tokenizer: BPETokenizer, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.last_metrics: Dict[str, Any] = {}

    def _estimate_kv_cache_size(self, kv_caches: Optional[List[Any]]) -> int:
        if not kv_caches:
            return 0
        total_bytes = 0
        for k, v in kv_caches:
            total_bytes += k.element_size() * k.nelement()
            total_bytes += v.element_size() * v.nelement()
        return total_bytes

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
        use_kv_cache: bool = True,
        streamer: Optional[BaseStreamer] = None
    ) -> Generator[str, None, None]:
        """Yields generated token strings sequentially while streaming to streamer callback."""
        stop_tokens = stop_tokens or [self.tokenizer.special_tokens.get("<eos>", 3)]
        
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not token_ids:
            return

        idx = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        prompt_len = idx.shape[1]
        
        start_time = time.perf_counter()
        first_token_time = None
        generated_count = 0
        kv_caches = None

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
            
            first_token_time = time.perf_counter()
            idx = torch.cat((idx, next_token), dim=1)
            generated_count += 1
            
            tok_id = next_token.item()
            tok_text = self.tokenizer.decode([tok_id], skip_special_tokens=False)
            if streamer:
                streamer.put(tok_id, tok_text)
            yield tok_text
            
            # 2. Incremental single-token decoding loop using past KV Caches
            while generated_count < max_new_tokens:
                if tok_id in stop_tokens:
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
                tok_id = next_token.item()
                tok_text = self.tokenizer.decode([tok_id], skip_special_tokens=False)
                
                if streamer:
                    streamer.put(tok_id, tok_text)
                yield tok_text

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

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                idx = torch.cat((idx, next_token), dim=1)
                generated_count += 1
                tok_id = next_token.item()
                tok_text = self.tokenizer.decode([tok_id], skip_special_tokens=False)
                
                if streamer:
                    streamer.put(tok_id, tok_text)
                yield tok_text

                if tok_id in stop_tokens:
                    break

        if streamer:
            streamer.end()

        end_time = time.perf_counter()
        elapsed_sec = end_time - start_time
        first_token_latency_ms = ((first_token_time - start_time) if first_token_time else elapsed_sec) * 1000.0
        tok_per_sec = generated_count / max(elapsed_sec, 1e-6)
        ms_per_tok = (elapsed_sec / max(generated_count, 1)) * 1000.0
        
        peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0
        cache_size_bytes = self._estimate_kv_cache_size(kv_caches)

        self.last_metrics = {
            "prompt_length": prompt_len,
            "tokens_generated": generated_count,
            "elapsed_sec": round(elapsed_sec, 4),
            "first_token_latency_ms": round(first_token_latency_ms, 2),
            "avg_latency_per_tok_ms": round(ms_per_tok, 2),
            "tokens_per_sec": round(tok_per_sec, 2),
            "peak_memory_mb": round(peak_mem_mb, 2),
            "cache_size_bytes": cache_size_bytes,
            "used_kv_cache": use_kv_cache
        }

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
        use_kv_cache: bool = True,
        draft_model: Optional[BaseModel] = None,
        k_draft: int = 4
    ) -> str:
        if draft_model is not None:
            from ai_core.inference.speculative import SpeculativeDecoder
            spec_decoder = SpeculativeDecoder(
                target_model=self.model,
                draft_model=draft_model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            res = spec_decoder.generate_speculative(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                k_draft=k_draft,
                temperature=temperature
            )
            self.last_metrics = res
            return res["text"]

        streamer = TokenStreamer()
        list(self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
            use_kv_cache=use_kv_cache,
            streamer=streamer
        ))
        return streamer.get_full_text()
