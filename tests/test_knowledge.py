import os
import pytest
from jarvis_core.knowledge.indexer import KnowledgeIndex
from jarvis_core.knowledge.file_watcher import FileWatcher

def test_knowledge_index_and_watcher(tmp_path):
    sample_file = str(tmp_path / "sample.py")
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write("def jarvis_kernel():\n    print('Hello Operating System')\n")

    index = KnowledgeIndex()
    watcher = FileWatcher(knowledge_index=index, extensions=[".py"])
    watcher.scan_directory(str(tmp_path))

    results = index.search("jarvis_kernel", top_k=2)
    assert len(results) > 0
    assert "jarvis_kernel" in results[0]["content"]
