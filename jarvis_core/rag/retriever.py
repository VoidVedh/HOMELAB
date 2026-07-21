from typing import List, Dict, Any, Optional
from jarvis_core.rag.query_rewriter import QueryRewriter
from jarvis_core.rag.reranker import Reranker

class Retriever:
    """Orchestrates query rewriting, candidate search against knowledge index, and reranking."""

    def __init__(self, knowledge_index: Any):
        self.knowledge_index = knowledge_index
        self.query_rewriter = QueryRewriter()
        self.reranker = Reranker()

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        rewritten_queries = self.query_rewriter.rewrite(query)
        
        candidates = []
        seen_ids = set()

        for q in rewritten_queries:
            results = self.knowledge_index.search(q, top_k=top_k * 2)
            for item in results:
                chunk_id = item.get("chunk_id")
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    candidates.append(item)

        return self.reranker.rerank(query=query, chunks=candidates, top_k=top_k)
