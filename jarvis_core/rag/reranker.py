from typing import List, Dict, Any

class Reranker:
    """Reranks retrieved document chunks based on query term overlap & recency."""

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        query_terms = set(query.lower().split())
        
        scored_chunks = []
        for chunk in chunks:
            text = chunk.get("content", "").lower()
            overlap_score = sum(1 for term in query_terms if term in text)
            
            # Recency bonus if timestamp present
            recency_score = chunk.get("timestamp", 0) / 1e10
            final_score = overlap_score + recency_score

            item = dict(chunk)
            item["score"] = final_score
            scored_chunks.append(item)

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
