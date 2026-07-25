from typing import List, Dict, Any, Optional
from jarvis_core.rag.bm25 import BM25Scorer
from jarvis_core.rag.cross_encoder import CrossEncoderReranker
from jarvis_core.rag.query_rewriter import QueryRewriter

class HybridRetriever:
    """
    Production Hybrid RAG Retriever combining Okapi BM25 Sparse Keyword Search, 
    Dense Vector Similarity Search, Reciprocal Rank Fusion (RRF), and Cross-Encoder Reranking.
    """

    def __init__(self, knowledge_index: Any, rrf_k: int = 60):
        self.knowledge_index = knowledge_index
        self.rrf_k = rrf_k
        self.bm25_scorer = BM25Scorer()
        self.cross_encoder = CrossEncoderReranker()
        self.query_rewriter = QueryRewriter()
        self.is_bm25_indexed = False

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Indexes documents into BM25 sparse engine."""
        self.bm25_scorer.fit(documents)
        self.is_bm25_indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes hybrid BM25 + dense retrieval, fuses rankings via Reciprocal Rank Fusion (RRF),
        and applies cross-encoder reranking.
        """
        rewritten_queries = self.query_rewriter.rewrite(query)

        # 1. Retrieve Dense Candidates from Knowledge Index
        dense_candidates = []
        seen_dense = set()
        for q in rewritten_queries:
            results = self.knowledge_index.search(q, top_k=top_k * 3)
            for item in results:
                cid = item.get("chunk_id", item.get("id"))
                if cid not in seen_dense:
                    seen_dense.add(cid)
                    dense_candidates.append(item)

        # Build BM25 index from knowledge index documents if not already indexed
        if not self.is_bm25_indexed and hasattr(self.knowledge_index, "chunks"):
            self.index_documents(self.knowledge_index.chunks)

        # 2. Retrieve Sparse BM25 Candidates
        bm25_candidates = self.bm25_scorer.search(query, top_k=top_k * 3)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        candidate_map: Dict[str, Dict[str, Any]] = {}

        # Accumulate Dense RRF scores
        for rank, item in enumerate(dense_candidates, start=1):
            cid = item.get("chunk_id", item.get("id", str(rank)))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            candidate_map[cid] = item

        # Accumulate BM25 RRF scores
        for rank, item in enumerate(bm25_candidates, start=1):
            cid = item.get("chunk_id", item.get("id", str(rank)))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            if cid not in candidate_map:
                candidate_map[cid] = item

        # Sort fused candidates by RRF score
        fused_list = []
        for cid, score in rrf_scores.items():
            item_copy = dict(candidate_map[cid])
            item_copy["rrf_score"] = round(score, 6)
            fused_list.append(item_copy)

        fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)

        # 4. Secondary Cross-Encoder Reranking
        top_fused = fused_list[:top_k * 2]
        reranked_final = self.cross_encoder.rerank(query=query, chunks=top_fused, top_k=top_k)

        return reranked_final
