import uuid
from typing import List, Dict, Any

class DocumentChunker:
    """Splits raw documents into sliding window chunks with overlap."""

    def chunk_document(
        self,
        file_path: str,
        content: str,
        chunk_size: int = 500,
        overlap: int = 100
    ) -> List[Dict[str, Any]]:
        if not content:
            return []

        chunks = []
        lines = content.splitlines()
        current_chunk_lines = []
        current_char_count = 0
        start_line = 1

        for idx, line in enumerate(lines, 1):
            current_chunk_lines.append(line)
            current_char_count += len(line) + 1

            if current_char_count >= chunk_size:
                chunk_str = "\n".join(current_chunk_lines)
                chunks.append({
                    "chunk_id": str(uuid.uuid4())[:8],
                    "file_path": file_path,
                    "lines": f"L{start_line}-L{idx}",
                    "content": chunk_str
                })
                # Retain overlap lines
                overlap_lines = current_chunk_lines[-(overlap // 20 + 1):]
                current_chunk_lines = list(overlap_lines)
                current_char_count = sum(len(l) + 1 for l in current_chunk_lines)
                start_line = max(1, idx - len(current_chunk_lines) + 1)

        if current_chunk_lines:
            chunk_str = "\n".join(current_chunk_lines)
            chunks.append({
                "chunk_id": str(uuid.uuid4())[:8],
                "file_path": file_path,
                "lines": f"L{start_line}-L{len(lines)}",
                "content": chunk_str
            })

        return chunks
