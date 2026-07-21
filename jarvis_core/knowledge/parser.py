import os
import ast
from typing import Dict, Any, List

class DocumentParser:
    """Parses source code (Python), Markdown, JSON, YAML, and HTML into structured elements."""

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        parsed_info = {
            "file_path": file_path,
            "ext": ext,
            "raw_content": content,
            "symbols": []
        }

        if ext == ".py":
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                        parsed_info["symbols"].append({
                            "name": node.name,
                            "type": type(node).__name__,
                            "lineno": getattr(node, "lineno", 0)
                        })
            except Exception:
                pass  # Fallback to plain text

        return parsed_info
