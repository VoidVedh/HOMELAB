import time
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from jarvis_core.sandbox.policy import ExecutionPolicy
from jarvis_core.sandbox.result import ExecutionResult

class BaseSandbox(ABC):
    """Abstract interface for execution sandboxes (Python, Bash, Docker, Git, Network, Remote)."""

    def __init__(self, default_policy: Optional[ExecutionPolicy] = None):
        self.default_policy = default_policy or ExecutionPolicy()

    @abstractmethod
    async def execute(self, code_or_command: str, policy: Optional[ExecutionPolicy] = None) -> ExecutionResult:
        """Executes target code/command using isolation policy."""
        pass

    @staticmethod
    async def _run_subprocess_async(
        cmd_args: list[str],
        policy: ExecutionPolicy,
        tool_name: str = "sandbox"
    ) -> ExecutionResult:
        t0 = time.perf_counter()
        
        try:
            policy.validate_command(cmd_args[0])
            env = dict(policy.environment) if policy.environment else None

            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=policy.working_directory,
                env=env
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=policy.timeout
                )
                timed_out = False
                exit_code = process.returncode if process.returncode is not None else 0
            except asyncio.TimeoutError:
                timed_out = True
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
                exit_code = -1

            duration_ms = (time.perf_counter() - t0) * 1000.0
            
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")[:policy.stdout_limit]
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")[:policy.stderr_limit]

            return ExecutionResult(
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=exit_code,
                duration_ms=round(duration_ms, 2),
                timed_out=timed_out,
                working_directory=policy.working_directory,
                command=" ".join(cmd_args),
                tool_name=tool_name,
                exception="TimeoutExceeded" if timed_out else None
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=1,
                duration_ms=round(duration_ms, 2),
                timed_out=False,
                working_directory=policy.working_directory,
                command=" ".join(cmd_args),
                tool_name=tool_name,
                exception=str(e)
            )
