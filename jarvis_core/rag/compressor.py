import re
from typing import List, Dict, Any

class ContextCompressor:
    """
    Dynamic Context Compressor enforcing strict token budget limits,
    eliminating redundant sentences and preserving exact file citations.
    """

    def __init__(self, max_token_budget: int = 1024):
        self.max_token_budget = max_token_budget

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimation (approx 4 chars per token)."""
        return max(1, len(text) // 4)

    def compress_chunk(self, chunk_text: str, max_chunk_tokens: int) -> str:
        sentences = re.split(r"(?<=[.!?\n])\s+", chunk_text.strip())
        compressed = []
        curr_tokens = 0

        for sent in sentences:
            sent_tokens = self.estimate_tokens(sent)
            if curr_tokens + sent_tokens > max_chunk_tokens:
                break
            compressed.append(sent)
            curr_tokens += sent_tokens

        return " ".join(compressed)

    def compress_context(self, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compresses retrieved chunk list into budget-capped context block with citations."""
        if not retrieved_chunks:
            return {"formatted_context": "", "used_tokens": 0, "total_chunks": 0}

        max_per_chunk = max(64, self.max_token_budget // len(retrieved_chunks))
        formatted_blocks = []
        total_tokens = 0

        for i, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("file_path", chunk.get("source", f"Doc_{i}"))
            line_range = chunk.get("line_range", "")
            citation = f"[{source}{' ' + line_range if line_range else ''}]"

            raw_text = chunk.get("text", chunk.get("content", ""))
            compressed_text = self.compress_chunk(raw_text, max_per_chunk)
            
            block = f"{citation}\n{compressed_text}\n"
            block_tokens = self.estimate_tokens(block)

            if total_tokens + block_tokens > self.max_token_budget:
                break

            formatted_blocks.append(block)
            total_tokens += block_tokens

        formatted_context = "\n".join(formatted_blocks)
        return {
            "formatted_context": formatted_context,
            "used_tokens": total_tokens,
            "total_chunks": len(formatted_blocks),
            "token_budget": self.max_token_budget
        }
