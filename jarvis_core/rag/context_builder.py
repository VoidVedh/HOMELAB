from typing import List, Dict, Any

class ContextBuilder:
    """Assembles ranked RAG document chunks into context windows adhering to token length limits."""

    def build_context_snippet(self, ranked_chunks: List[Dict[str, Any]], max_chars: int = 2000) -> str:
        if not ranked_chunks:
            return ""

        snippets = []
        current_len = 0

        for chunk in ranked_chunks:
            path = chunk.get("file_path", "unknown")
            lines = chunk.get("lines", "")
            content = chunk.get("content", "").strip()
            
            formatted = f"--- [File: {path} {lines}] ---\n{content}\n"
            if current_len + len(formatted) > max_chars:
                break
            
            snippets.append(formatted)
            current_len += len(formatted)

        return "\n".join(snippets)
