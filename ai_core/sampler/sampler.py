import torch
import torch.nn.functional as F

class LogitSampler:
    """Logit Sampler supporting Temperature scaling, Top-K, and Top-P (Nucleus) sampling."""
    
    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        greedy: bool = False
    ) -> torch.Tensor:
        """
        logits: (batch_size, vocab_size) tensor of raw logit scores for the last token.
        Returns tensor of shape (batch_size, 1) containing sampled token IDs.
        """
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
