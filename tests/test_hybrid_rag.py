import pytest
from jarvis_core.rag.bm25 import BM25Scorer
from jarvis_core.rag.cross_encoder import CrossEncoderReranker
from jarvis_core.rag.hybrid_retriever import HybridRetriever
from jarvis_core.rag.compressor import ContextCompressor

class MockKnowledgeIndex:
    def __init__(self):
        self.chunks = [
            {"chunk_id": "c1", "text": "def quantize_fp32_to_nf4(weight):\n    # NormalFloat 4-bit quantization\n    return packed, scales", "file_path": "ai_core/quantization/quant_utils.py"},
            {"chunk_id": "c2", "text": "class QuantizedLinear(nn.Module):\n    # Linear projection layer with 4-bit weights\n    pass", "file_path": "ai_core/quantization/quantized_linear.py"},
            {"chunk_id": "c3", "text": "class ExecutivePlanner:\n    # Task DAG planner for execution", "file_path": "jarvis_core/planner/executive_planner.py"}
        ]

    def search(self, query: str, top_k: int = 5):
        return [self.chunks[0], self.chunks[1]]

def test_bm25_scorer_search():
    docs = [
        {"id": "d1", "text": "QuantizedLinear layer for 4-bit NormalFloat NF4 weight quantization."},
        {"id": "d2", "text": "ExecutivePlanner generates Task DAG for agent execution."},
        {"id": "d3", "text": "SpeculativeDecoder uses draft model to accelerate inference."}
    ]

    bm25 = BM25Scorer(k1=1.5, b=0.75)
    bm25.fit(docs)
    
    results = bm25.search("QuantizedLinear 4-bit", top_k=2)
    assert len(results) > 0
    assert results[0]["id"] == "d1"
    assert results[0]["bm25_score"] > 0.0

def test_cross_encoder_reranker():
    reranker = CrossEncoderReranker()
    chunks = [
        {"id": "1", "text": "Unrelated documentation text."},
        {"id": "2", "text": "Exact match for QuantizedLinear implementation with NF4 weights."}
    ]

    reranked = reranker.rerank("QuantizedLinear", chunks, top_k=2)
    assert reranked[0]["id"] == "2"
    assert reranked[0]["cross_encoder_score"] > reranked[1]["cross_encoder_score"]

def test_hybrid_retriever_rrf_fusion():
    k_index = MockKnowledgeIndex()
    retriever = HybridRetriever(knowledge_index=k_index, rrf_k=60)
    retriever.index_documents(k_index.chunks)

    results = retriever.retrieve("quantize_fp32_to_nf4", top_k=2)
    assert len(results) > 0
    assert "rrf_score" in results[0]
    assert "cross_encoder_score" in results[0]

def test_context_compressor_budget():
    compressor = ContextCompressor(max_token_budget=100)
    chunks = [
        {"text": "A " * 200, "file_path": "file1.py", "line_range": "L1-L20"},
        {"text": "B " * 200, "file_path": "file2.py", "line_range": "L10-L40"}
    ]

    res = compressor.compress_context(chunks)
    assert res["used_tokens"] <= 100
    assert "[file1.py L1-L20]" in res["formatted_context"]
