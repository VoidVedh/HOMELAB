import time
from typing import List, Dict, Any
from jarvis_core.knowledge.parser import DocumentParser
from jarvis_core.knowledge.chunker import DocumentChunker

class KnowledgeIndex:
    """In-memory & persistent knowledge index storing file document chunks."""

    def __init__(self):
        self._chunks: Dict[str, Dict[str, Any]] = {}
        self.parser = DocumentParser()
        self.chunker = DocumentChunker()

    def index_file(self, file_path: str):
        parsed = self.parser.parse_file(file_path)
        chunks = self.chunker.chunk_document(file_path, parsed["raw_content"])
        
        # Remove old chunks for this file
        self._chunks = {cid: c for cid, c in self._chunks.items() if c["file_path"] != file_path}
        
        # Add new chunks
        for c in chunks:
            c["timestamp"] = time.time()
            self._chunks[c["chunk_id"]] = c

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_terms = set(query.lower().split())
        results = []

        for chunk in self._chunks.values():
            text = chunk["content"].lower()
            score = sum(1 for t in query_terms if t in text)
            if score > 0:
                item = dict(chunk)
                item["score"] = score
                results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
