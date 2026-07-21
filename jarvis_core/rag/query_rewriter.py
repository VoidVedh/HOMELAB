import re
from typing import List

class QueryRewriter:
    """Rewrites and expands user queries for improved retrieval relevance."""

    def rewrite(self, query: str) -> List[str]:
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower()).strip()
        words = [w for w in cleaned.split() if len(w) > 2]
        
        expanded = [query]
        if words:
            expanded.append(" ".join(words))
            # Add key noun / term extractions
            keywords = [w for w in words if w not in {"the", "and", "for", "with", "this", "that", "how", "what", "can"}]
            if keywords:
                expanded.append(" ".join(keywords))
        
        return list(dict.fromkeys(expanded))
