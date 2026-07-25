from typing import Optional

from jarvis_core.sandbox.base_sandbox import BaseSandbox
from jarvis_core.sandbox.policy import ExecutionPolicy
from jarvis_core.sandbox.result import ExecutionResult

class BashSandbox(BaseSandbox):
    """Isolated Bash shell command execution sandbox."""

    async def execute(self, code_or_command: str, policy: Optional[ExecutionPolicy] = None) -> ExecutionResult:
        policy = policy or self.default_policy

        cmd = ["bash", "-c", code_or_command]
        res = await self._run_subprocess_async(cmd, policy=policy, tool_name="BashSandbox")
        return res
