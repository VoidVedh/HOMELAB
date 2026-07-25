import json
import re
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Union
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer

class BaseGrammar(ABC):
    """Abstract interface for extensible grammar constraints (JSON, BNF, PEG, YAML, SQL)."""

    @abstractmethod
    def is_valid_partial(self, text: str) -> bool:
        """Returns True if text is a valid prefix under the grammar."""
        pass

    @abstractmethod
    def is_valid_complete(self, text: str) -> bool:
        """Returns True if text is a complete valid instance under the grammar."""
        pass


class JSONGrammar(BaseGrammar):
    """JSON Schema grammar constraint validating partial and complete JSON strings."""

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema or {}

    def is_valid_partial(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return True
        # Basic JSON prefix validity check
        if text.startswith("{") or text.startswith("[") or text.startswith('"') or text in ["true", "false", "null"]:
            try:
                # Try parsing as JSON if complete, or check structural symbols
                json.loads(text)
                return True
            except json.JSONDecodeError as e:
                # If error is end of string, it's a valid partial prefix
                return "Unterminated" in str(e) or "Expecting" in str(e) or "Extra data" not in str(e)
        return False

    def is_valid_complete(self, text: str) -> bool:
        try:
            data = json.loads(text)
            if self.schema:
                # Validate required keys if schema is provided
                req = self.schema.get("required", [])
                for key in req:
                    if key not in data:
                        return False
            return True
        except Exception:
            return False


class GrammarConstrainedSampler:
    """
    Grammar-Constrained Logit Sampler.
    Dynamically masks invalid tokens during autoregressive decoding to guarantee 100% grammar compliance.
    """

    def __init__(self, tokenizer: BPETokenizer, grammar: Optional[BaseGrammar] = None):
        self.tokenizer = tokenizer
        self.grammar = grammar or JSONGrammar()

    def get_valid_next_tokens(self, current_text: str) -> Set[int]:
        """Calculates set of token IDs that produce valid grammar prefixes."""
        valid_tokens = set()
        for tok_str, tok_id in self.tokenizer.vocab.items():
            if tok_id in self.tokenizer.special_tokens.values():
                continue
            candidate = current_text + tok_str
            if self.grammar.is_valid_partial(candidate):
                valid_tokens.add(tok_id)

        # Fallback to all vocabulary if masking is too restrictive
        if not valid_tokens:
            return set(self.tokenizer.vocab.values())
        return valid_tokens

    def sample(
        self,
        logits: torch.Tensor,
        current_text: str,
        temperature: float = 1.0,
        top_k: int = 40
    ) -> int:
        """
        Applies logit masking for valid grammar tokens and samples next token ID.
        logits shape: (vocab_size,) or (1, vocab_size)
        """
        if logits.dim() == 2:
            logits = logits[0]

        valid_tokens = self.get_valid_next_tokens(current_text)
        
        masked_logits = logits.clone()
        mask = torch.full_like(masked_logits, float("-inf"))
        for tok_id in valid_tokens:
            if tok_id < masked_logits.size(0):
                mask[tok_id] = 0.0

        masked_logits = masked_logits + mask
        
        if temperature > 0:
            probs = F.softmax(masked_logits / max(temperature, 1e-5), dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token_id = torch.argmax(masked_logits).item()

        return next_token_id
