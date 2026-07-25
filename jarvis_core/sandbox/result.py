from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class ExecutionResult:
    """Standardized result returned by all sandbox execution targets."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0
    cpu_time: float = 0.0
    memory_usage: int = 0
    timed_out: bool = False
    working_directory: str = "."
    command: str = ""
    tool_name: str = "sandbox"
    artifacts: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    exception: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.exception is None
