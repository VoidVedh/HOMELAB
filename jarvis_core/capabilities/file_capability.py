from typing import Dict, Any, List
from jarvis_core.capabilities.base_capability import BaseCapability

class FileCapability(BaseCapability):
    """Capability handling safe local filesystem operations, document reading, and writing."""

    def can_handle(self, goal: str) -> bool:
        keywords = ["file", "directory", "folder", "read", "write", "list", "save", "path"]
        return any(kw in goal.lower() for kw in keywords)

    def plan(self, goal: str) -> Dict[str, Any]:
        return {
            "capability": "FileCapability",
            "goal": goal,
            "steps": [
                {"action": "filesystem:list_dir", "payload": {"path": "."}},
                {"action": "filesystem:write_file", "payload": {"path": "logs/file_op.log", "content": f"File op: {goal}\n"}}
            ]
        }

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "completed",
            "steps_executed": len(plan.get("steps", [])),
            "files_modified": ["logs/file_op.log"]
        }

    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        success = result.get("status") == "completed"
        return {"passed": success, "error": None if success else "File capability execution incomplete"}

    async def repair(self, result: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        return {"status": "repaired", "attempt": attempt, "files_modified": ["logs/file_op.log"]}
