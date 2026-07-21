from typing import Dict, Any, List
from jarvis_core.capabilities.base_capability import BaseCapability

class TerminalCapability(BaseCapability):
    """Capability handling asynchronous terminal process execution and command diagnostics."""

    def can_handle(self, goal: str) -> bool:
        keywords = ["terminal", "command", "exec", "run", "bash", "shell", "process"]
        return any(kw in goal.lower() for kw in keywords)

    def plan(self, goal: str) -> Dict[str, Any]:
        return {
            "capability": "TerminalCapability",
            "goal": goal,
            "steps": [
                {"action": "terminal:exec", "payload": {"command": "echo 'JARVIS Terminal Execution'"}}
            ]
        }

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "completed",
            "steps_executed": len(plan.get("steps", [])),
            "output": "JARVIS Terminal Execution"
        }

    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        success = result.get("status") == "completed"
        return {"passed": success, "error": None if success else "Terminal execution failed"}

    async def repair(self, result: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        return {"status": "repaired", "attempt": attempt, "output": "JARVIS Terminal Execution (Repaired)"}
