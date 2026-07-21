import json
import time
import os
from typing import Dict, Any, List

class StructuredLogger:
    """Emits structured JSON execution telemetry records."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_execution(
        self,
        goal: str,
        task_id: str,
        plan_steps: int,
        tools_used: List[str],
        duration_ms: float,
        success: bool,
        repair_attempts: int = 0,
        errors: List[str] = None
    ) -> Dict[str, Any]:
        record = {
            "timestamp": time.time(),
            "goal": goal,
            "task_id": task_id,
            "plan_steps": plan_steps,
            "tools_used": tools_used,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "repair_attempts": repair_attempts,
            "errors": errors or []
        }

        log_file = os.path.join(self.log_dir, "execution_telemetry.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record
