import torch
from typing import Optional, List
from ai_core.transformer.base_model import BaseModel
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.sampler.sampler import LogitSampler
from ai_core.utils.device import Timer, get_device

class InferenceEngine:
    """Autoregressive text generation engine for BaseModel instances."""
    def __init__(self, model: BaseModel, tokenizer: BPETokenizer, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None
    ) -> str:
        stop_tokens = stop_tokens or [self.tokenizer.special_tokens.get("<eos>", 3)]
        
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        idx = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with Timer() as timer:
            for _ in range(max_new_tokens):
                # Truncate to max sequence length if necessary
                idx_cond = idx[:, -self.model.config.max_seq_len:]
                logits, _ = self.model(idx_cond)
                
                # Get last token logits
                last_logits = logits[:, -1, :]
                
                next_token = LogitSampler.sample(
                    logits=last_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                idx = torch.cat((idx, next_token), dim=1)
                
                if next_token.item() in stop_tokens:
                    break

        generated_tokens = idx[0].cpu().tolist()
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
