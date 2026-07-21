import os
import glob
from typing import Dict, Any, List
from plugins.base_plugin import BasePlugin

class FilesystemPlugin(BasePlugin):
    """Plugin exposing safe filesystem operations to JARVIS planner."""

    def name(self) -> str:
        return "filesystem"

    def description(self) -> str:
        return "Reads, writes, lists, and manages local files and directories."

    def permissions(self) -> List[str]:
        return ["read_file", "write_file", "list_dir"]

    async def run(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if action == "read_file":
            path = payload.get("path")
            if not os.path.exists(path):
                return {"error": f"File not found: {path}"}
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return {"path": path, "content": content}

        elif action == "write_file":
            path = payload.get("path")
            content = payload.get("content", "")
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"path": path, "status": "written", "bytes": len(content)}

        elif action == "list_dir":
            path = payload.get("path", ".")
            if not os.path.exists(path):
                return {"error": f"Directory not found: {path}"}
            items = os.listdir(path)
            return {"path": path, "items": items}

        return {"error": f"Unknown filesystem action: {action}"}
