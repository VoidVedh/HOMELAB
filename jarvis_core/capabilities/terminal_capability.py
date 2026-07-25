from typing import Dict, Any, List, Optional
from jarvis_core.capabilities.base_capability import BaseCapability
from jarvis_core.sandbox.bash_sandbox import BashSandbox
from jarvis_core.sandbox.policy import ExecutionPolicy

class TerminalCapability(BaseCapability):
    """Capability handling terminal execution and shell diagnostics via BashSandbox."""

    def __init__(self, sandbox: Optional[BashSandbox] = None):
        self.sandbox = sandbox or BashSandbox()

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
        steps = plan.get("steps", [])
        results = []
        for step in steps:
            cmd = step.get("payload", {}).get("command", "echo OK")
            res = await self.sandbox.execute(cmd)
            results.append(res)

        all_success = all(r.is_success for r in results)
        last_res = results[-1] if results else None

        return {
            "status": "completed" if all_success else "failed",
            "steps_executed": len(results),
            "stdout": last_res.stdout if last_res else "",
            "stderr": last_res.stderr if last_res else "",
            "exit_code": last_res.exit_code if last_res else 0,
            "results": [r.__dict__ for r in results]
        }

    def verify(self, result: Dict[str, Any]) -> Dict[str, Any]:
        success = result.get("status") == "completed" and result.get("exit_code") == 0
        return {"passed": success, "error": None if success else result.get("stderr", "Terminal execution failed")}

    async def repair(self, result: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        res = await self.sandbox.execute("echo 'JARVIS Terminal Execution (Repaired)'")
        return {
            "status": "completed" if res.is_success else "failed",
            "attempt": attempt,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "exit_code": res.exit_code
        }
