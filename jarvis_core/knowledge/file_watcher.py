import os
import time
from typing import Dict, List
from jarvis_core.knowledge.indexer import KnowledgeIndex

class FileWatcher:
    """Workspace file scanner updating KnowledgeIndex dynamically when files change."""

    def __init__(self, knowledge_index: KnowledgeIndex, extensions: List[str] = None):
        self.knowledge_index = knowledge_index
        self.extensions = set(extensions or [".md", ".py", ".json", ".yaml", ".html", ".txt"])
        self._mtimes: Dict[str, float] = {}

    def scan_directory(self, root_dir: str):
        for root, dirs, files in os.walk(root_dir):
            if "venv" in root or ".git" in root or "__pycache__" in root:
                continue
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.extensions:
                    file_path = os.path.join(root, file)
                    mtime = os.path.getmtime(file_path)
                    if file_path not in self._mtimes or self._mtimes[file_path] < mtime:
                        self.knowledge_index.index_file(file_path)
                        self._mtimes[file_path] = mtime
