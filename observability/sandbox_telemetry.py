import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

from jarvis_core.sandbox.result import ExecutionResult

class SandboxTelemetryLogger:
    """Structured telemetry logger persisting sandboxed execution metrics under observability/telemetry/."""

    def __init__(self, telemetry_dir: str = "observability/telemetry"):
        self.telemetry_dir = telemetry_dir
        os.makedirs(self.telemetry_dir, exist_ok=True)
        self.log_file = os.path.join(self.telemetry_dir, f"sandbox_trace_{int(time.time())}.jsonl")

    def log_execution(
        self,
        result: ExecutionResult,
        repair_attempts: int = 0,
        verification_passed: bool = True
    ) -> Dict[str, Any]:
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": result.tool_name,
            "command": result.command,
            "duration_ms": result.duration_ms,
            "cpu_time": result.cpu_time,
            "memory_usage": result.memory_usage,
            "stdout_bytes": len(result.stdout.encode("utf-8")),
            "stderr_bytes": len(result.stderr.encode("utf-8")),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "repair_attempts": repair_attempts,
            "verification_passed": verification_passed
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return entry
