import logging
from typing import Dict, Any, List, Optional
from ai_core.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

import logging
from typing import Dict, Any, List, Optional
from ai_core.inference.engine import InferenceEngine
from observability.sandbox_telemetry import SandboxTelemetryLogger

logger = logging.getLogger(__name__)

class RepairEngine:
    """Manages critique and automated repair loop up to max 3 attempts capturing stdout/stderr/exit_code."""

    def __init__(
        self,
        inference_engine: Optional[InferenceEngine] = None,
        max_attempts: int = 3,
        telemetry_logger: Optional[SandboxTelemetryLogger] = None
    ):
        self.inference_engine = inference_engine
        self.max_attempts = max_attempts
        self.telemetry = telemetry_logger or SandboxTelemetryLogger()

    def classify_failure(self, stderr: str, exit_code: int) -> str:
        if exit_code == -1:
            return "TIMEOUT_EXCEEDED"
        if "SyntaxError" in stderr or "IndentationError" in stderr:
            return "SYNTAX_ERROR"
        if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
            return "IMPORT_ERROR"
        if "PermissionError" in stderr:
            return "PERMISSION_ERROR"
        return "RUNTIME_ERROR"

    def critique_failure(self, task_id: str, error_msg: str, exit_code: int = 1) -> Dict[str, Any]:
        failure_type = self.classify_failure(error_msg, exit_code)
        return {
            "task_id": task_id,
            "failure_type": failure_type,
            "error_summary": error_msg,
            "suggested_fix": f"[{failure_type}] Re-evaluate script or command inputs: {error_msg[:100]}"
        }

    async def attempt_repair(
        self,
        task_id: str,
        error_msg: str,
        current_attempt: int,
        execute_fn,
        exit_code: int = 1
    ) -> Dict[str, Any]:
        if current_attempt > self.max_attempts:
            return {
                "repaired": False,
                "reason": f"Exceeded maximum repair attempts ({self.max_attempts})"
            }

        critique = self.critique_failure(task_id, error_msg, exit_code=exit_code)
        logger.info(f"[*] [RepairEngine] Attempt {current_attempt}/{self.max_attempts} for task '{task_id}': {critique['suggested_fix']}")

        # Re-execute via callback
        try:
            repair_result = await execute_fn()
            
            repaired = repair_result.get("status") in ["completed", "repaired", "success"] or repair_result.get("exit_code") == 0
            
            return {
                "repaired": repaired,
                "attempt": current_attempt,
                "result": repair_result,
                "critique": critique
            }
        except Exception as e:
            return {
                "repaired": False,
                "attempt": current_attempt,
                "error": str(e),
                "critique": critique
            }
