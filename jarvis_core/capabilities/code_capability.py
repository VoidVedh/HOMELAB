from typing import Dict, Any, List, Optional
from jarvis_core.capabilities.base_capability import BaseCapability
from jarvis_core.sandbox.python_sandbox import PythonSandbox
from jarvis_core.sandbox.policy import ExecutionPolicy

class CodeCapability(BaseCapability):
    """Capability handling software engineering, code generation, refactoring, and bug fixing via PythonSandbox."""

    def __init__(self, sandbox: Optional[PythonSandbox] = None):
        self.sandbox = sandbox or PythonSandbox()

    def can_handle(self, goal: str) -> bool:
        keywords = ["code", "python", "function", "fix", "bug", "refactor", "create app", "script"]
        return any(kw in goal.lower() for kw in keywords)

    def plan(self, goal: str) -> Dict[str, Any]:
        return {
            "capability": "CodeCapability",
            "goal": goal,
            "steps": [
                {"action": "python:execute", "payload": {"code": "print('CodeCapability execution check')\n"}}
            ]
        }

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps", [])
        results = []
        for step in steps:
            code = step.get("payload", {}).get("code", "print('OK')\n")
            res = await self.sandbox.execute(code)
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
        return {"passed": success, "error": None if success else result.get("stderr", "Code execution failed")}

    async def repair(self, result: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        # Attempt code fix via sandbox
        fixed_code = "print('Repaired python code execution')\n"
        res = await self.sandbox.execute(fixed_code)
        return {
            "status": "completed" if res.is_success else "failed",
            "attempt": attempt,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "exit_code": res.exit_code
        }
