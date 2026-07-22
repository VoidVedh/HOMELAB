import torch
import torch.nn.functional as F
from typing import Optional

class LogitSampler:
    """Logit Sampler supporting Temperature scaling, Top-K, Top-P (Nucleus), and Repetition Penalties."""
    
    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        greedy: bool = False,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        logits: (batch_size, vocab_size) tensor of raw logit scores for the last token.
        input_ids: (batch_size, seq_len) tensor of previously generated tokens for penalty calculations.
        Returns tensor of shape (batch_size, 1) containing sampled token IDs.
        """
        logits = logits.clone()

        if input_ids is not None and (repetition_penalty != 1.0 or frequency_penalty != 0.0 or presence_penalty != 0.0):
            batch_size, _ = logits.shape
            for b in range(batch_size):
                counts = torch.bincount(input_ids[b], minlength=logits.shape[-1])
                nonzero_tokens = torch.nonzero(counts).squeeze(-1)
                
                if repetition_penalty != 1.0 and nonzero_tokens.numel() > 0:
                    for token_id in nonzero_tokens:
                        tid = token_id.item()
                        if logits[b, tid] > 0:
                            logits[b, tid] /= repetition_penalty
                        else:
                            logits[b, tid] *= repetition_penalty
                            
                if frequency_penalty != 0.0 or presence_penalty != 0.0:
                    logits[b] -= (counts.float() * frequency_penalty + (counts > 0).float() * presence_penalty)

        if greedy or temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1:]
            logits[indices_to_remove] = float("-inf")

        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift indices to keep at least one token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        return next_tokens
