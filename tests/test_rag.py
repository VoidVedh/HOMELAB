import pytest
from jarvis_core.rag.query_rewriter import QueryRewriter
from jarvis_core.rag.reranker import Reranker
from jarvis_core.rag.context_builder import ContextBuilder

def test_query_rewriter():
    rewriter = QueryRewriter()
    queries = rewriter.rewrite("How to build a custom transformer model?")
    assert len(queries) >= 1
    assert "How to build a custom transformer model?" in queries

def test_reranker_and_context_builder():
    reranker = Reranker()
    builder = ContextBuilder()

    chunks = [
        {"chunk_id": "c1", "file_path": "a.txt", "lines": "L1-L5", "content": "Transformer architecture details", "timestamp": 100},
        {"chunk_id": "c2", "file_path": "b.txt", "lines": "L1-L5", "content": "Unrelated python code", "timestamp": 50}
    ]

    ranked = reranker.rerank("Transformer architecture", chunks, top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["chunk_id"] == "c1"

    snippet = builder.build_context_snippet(ranked, max_chars=1000)
    assert "Transformer architecture details" in snippet
