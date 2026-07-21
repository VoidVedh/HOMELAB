from typing import Dict, Any, List
from jarvis_core.capabilities.base_capability import BaseCapability

class CodeCapability(BaseCapability):
    """Capability handling software engineering, code generation, refactoring, and bug fixing."""

    def can_handle(self, goal: str) -> bool:
        keywords = ["code", "python", "function", "fix", "bug", "refactor", "create app", "script"]
        return any(kw in goal.lower() for kw in keywords)

    def plan(self, goal: str) -> Dict[str, Any]:
        return {
            "capability": "CodeCapability",
            "goal": goal,
            "steps": [
                {"action": "filesystem:write_file", "payload": {"path": "src/main.py", "content": "# Generated code\n"}},
                {"action": "terminal:exec", "payload": {"command": "python3 -m py_compile src/main.py"}}
            ]
        }

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "completed",
            "steps_executed": len(plan.get("steps", [])),
            "files_modified": ["src/main.py"]
        }

    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        success = result.get("status") == "completed"
        return {"passed": success, "error": None if success else "Execution status incomplete"}

    async def repair(self, result: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        return {"status": "repaired", "attempt": attempt, "files_modified": ["src/main.py"]}
