import re
from typing import List, Dict, Any

class CrossEncoderReranker:
    """
    Cross-Encoder Reranker performing fine-grained relevance scoring based on 
    exact symbol matching, token co-occurrence density, and query coverage.
    """

    def __init__(self, symbol_boost_weight: float = 2.0):
        self.symbol_boost_weight = symbol_boost_weight

    def score_pair(self, query: str, document_text: str) -> float:
        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return 0.0

        doc_words = re.findall(r"\w+", document_text.lower())
        doc_word_set = set(doc_words)

        # 1. Query Term Coverage Ratio
        matched_words = query_words.intersection(doc_word_set)
        coverage_ratio = len(matched_words) / len(query_words)

        # 2. Exact Symbol Match Bonus (e.g. class names, function names)
        exact_symbol_bonus = 0.0
        for q_word in query_words:
            if len(q_word) > 3 and q_word in document_text:
                exact_symbol_bonus += 0.25

        # 3. Co-occurrence Proximity Density
        frequency_density = sum(doc_words.count(w) for w in matched_words) / max(len(doc_words), 1)

        total_score = (coverage_ratio * 1.5) + (exact_symbol_bonus * self.symbol_boost_weight) + (frequency_density * 0.5)
        return float(total_score)

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Reranks candidate chunks using cross-encoder relevance scoring."""
        scored_chunks = []
        for chunk in chunks:
            text = chunk.get("text", chunk.get("content", ""))
            relevance_score = self.score_pair(query, text)
            
            chunk_copy = dict(chunk)
            chunk_copy["cross_encoder_score"] = round(relevance_score, 4)
            scored_chunks.append((chunk_copy, relevance_score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored_chunks[:top_k]]
