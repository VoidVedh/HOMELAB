import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class ExecutionPolicy:
    """Execution policy defining resource limits, permissions, and security constraints for sandboxed processes."""

    timeout: float = 10.0  # Seconds
    cpu_limit: Optional[float] = None
    memory_limit: Optional[int] = 512 * 1024 * 1024  # 512 MB
    stdout_limit: int = 100 * 1024  # 100 KB
    stderr_limit: int = 100 * 1024  # 100 KB
    working_directory: str = "."
    allowed_directories: List[str] = field(default_factory=lambda: ["."])
    allowed_commands: List[str] = field(default_factory=lambda: ["python3", "python3.14", "python", "bash", "ls", "echo", "cat", "pytest"])
    environment: Dict[str, str] = field(default_factory=dict)
    network_enabled: bool = False
    filesystem_mode: str = "read_write"  # "read_only" or "read_write"
    allow_write: bool = True
    allow_execute: bool = True

    def validate_command(self, command_str: str):
        """Prevents command injection and disallowed commands."""
        if not command_str or not command_str.strip():
            raise ValueError("Empty command string provided to ExecutionPolicy")
        
        cmd_head = command_str.strip().split()[0]
        base_cmd = os.path.basename(cmd_head)
        
        if self.allowed_commands and base_cmd not in self.allowed_commands and cmd_head not in self.allowed_commands:
            raise PermissionError(f"Command '{base_cmd}' is disallowed by ExecutionPolicy. Allowed: {self.allowed_commands}")
