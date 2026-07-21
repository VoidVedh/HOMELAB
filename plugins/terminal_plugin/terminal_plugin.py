import asyncio
from typing import Dict, Any, List
from plugins.base_plugin import BasePlugin

class TerminalPlugin(BasePlugin):
    """Plugin executing shell commands asynchronously with output capture."""

    def name(self) -> str:
        return "terminal"

    def description(self) -> str:
        return "Executes shell commands, captures stdout/stderr, and returns exit codes."

    def permissions(self) -> List[str]:
        return ["execute_command"]

    async def run(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if action == "exec":
            cmd = payload.get("command")
            cwd = payload.get("cwd", ".")
            if not cmd:
                return {"error": "No command provided"}

            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            return {
                "command": cmd,
                "exit_code": proc.returncode,
                "stdout": stdout.decode("utf-8", errors="ignore"),
                "stderr": stderr.decode("utf-8", errors="ignore")
            }

        return {"error": f"Unknown terminal action: {action}"}
