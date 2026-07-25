import os
import sys
import tempfile
from typing import Optional

from jarvis_core.sandbox.base_sandbox import BaseSandbox
from jarvis_core.sandbox.policy import ExecutionPolicy
from jarvis_core.sandbox.result import ExecutionResult

class PythonSandbox(BaseSandbox):
    """Isolated Python code execution sandbox."""

    async def execute(self, code_or_command: str, policy: Optional[ExecutionPolicy] = None) -> ExecutionResult:
        policy = policy or self.default_policy

        # Create temporary script file for execution
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(code_or_command)
            tmp_path = tmp.name

        try:
            cmd = [sys.executable, tmp_path]
            res = await self._run_subprocess_async(cmd, policy=policy, tool_name="PythonSandbox")
            return res
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
