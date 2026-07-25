import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

from ai_core.transformer.base_model import BaseModel
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer

class SpeculativeDecoder:
    """
    Speculative Decoding Engine using a fast draft model (Jarvis-0) to propose K candidate tokens 
    verified in parallel by a target model (Jarvis-1) with rejection sampling.
    """

    def __init__(
        self,
        target_model: BaseModel,
        draft_model: BaseModel,
        tokenizer: BPETokenizer,
        device: Optional[torch.device] = None
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")

        self.target_model.to(self.device).eval()
        self.draft_model.to(self.device).eval()

    @torch.no_grad()
    def generate_speculative(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        k_draft: int = 4,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        prompt_len = current_ids.size(1)

        total_accepted = 0
        total_proposed = 0
        forward_passes = 0

        while current_ids.size(1) - prompt_len < max_new_tokens:
            # 1. Propose K draft tokens autoregressively using draft_model
            draft_tokens = []
            draft_probs_list = []
            
            draft_ctx = current_ids.clone()
            for _ in range(k_draft):
                if draft_ctx.size(1) - prompt_len >= max_new_tokens:
                    break
                logits, _ = self.draft_model(draft_ctx)
                next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
                probs = F.softmax(next_token_logits, dim=-1)
                
                next_tok = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(next_tok.item())
                draft_probs_list.append(probs[0, next_tok.item()].item())
                
                draft_ctx = torch.cat([draft_ctx, next_tok], dim=1)

            if not draft_tokens:
                break

            actual_k = len(draft_tokens)
            total_proposed += actual_k

            # 2. Parallel target model forward pass over prompt + draft tokens
            target_logits, _ = self.target_model(draft_ctx)
            forward_passes += 1

            # 3. Rejection Sampling Verification
            n_accepted = 0
            for i in range(actual_k):
                token_id = draft_tokens[i]
                p_draft = draft_probs_list[i]
                
                # Target probability for the draft token at position (prompt_len + current_offset + i)
                target_pos_logits = target_logits[0, current_ids.size(1) - 1 + i, :] / max(temperature, 1e-5)
                target_probs = F.softmax(target_pos_logits, dim=-1)
                p_target = target_probs[token_id].item()

                # Accept token if random < min(1, p_target / p_draft)
                ratio = p_target / max(p_draft, 1e-8)
                r = torch.rand(1).item()
                
                if r < min(1.0, ratio):
                    n_accepted += 1
                else:
                    # Token rejected! Sample replacement token from max(0, P_target - P_draft)
                    adj_probs = torch.relu(target_probs - F.softmax(logits[:, -1, :], dim=-1)[0])
                    if adj_probs.sum() > 0:
                        adj_probs = adj_probs / adj_probs.sum()
                        replacement_tok = torch.multinomial(adj_probs, num_samples=1)
                    else:
                        replacement_tok = torch.multinomial(target_probs, num_samples=1)

                    # Append accepted tokens up to i + replacement token
                    accepted_tensor = torch.tensor([[draft_tokens[j] for j in range(i)] + [replacement_tok.item()]], device=self.device)
                    current_ids = torch.cat([current_ids, accepted_tensor], dim=1)
                    break
            else:
                # All K draft tokens accepted! Append draft tokens + 1 bonus token from target model
                last_target_logits = target_logits[0, -1, :] / max(temperature, 1e-5)
                last_target_probs = F.softmax(last_target_logits, dim=-1)
                bonus_tok = torch.multinomial(last_target_probs, num_samples=1)
                
                accepted_tensor = torch.tensor([[t for t in draft_tokens] + [bonus_tok.item()]], device=self.device)
                current_ids = torch.cat([current_ids, accepted_tensor], dim=1)
                n_accepted = actual_k

            total_accepted += n_accepted

        elapsed = time.perf_counter() - start_time
        gen_tokens = current_ids[0, prompt_len:].tolist()
        generated_text = self.tokenizer.decode(gen_tokens)

        acceptance_rate = total_accepted / max(total_proposed, 1)
        tokens_gen = len(gen_tokens)
        tok_per_sec = tokens_gen / max(elapsed, 1e-6)

        return {
            "text": generated_text,
            "tokens": gen_tokens,
            "num_generated_tokens": tokens_gen,
            "total_proposed": total_proposed,
            "total_accepted": total_accepted,
            "acceptance_rate_pct": round(acceptance_rate * 100.0, 2),
            "forward_passes": forward_passes,
            "elapsed_sec": round(elapsed, 4),
            "tokens_per_sec": round(tok_per_sec, 2)
        }
