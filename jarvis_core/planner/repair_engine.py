import logging
from typing import Dict, Any, List, Optional
from ai_core.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

class RepairEngine:
    """Manages critique and automated repair loop up to max 3 attempts."""

    def __init__(self, inference_engine: Optional[InferenceEngine] = None, max_attempts: int = 3):
        self.inference_engine = inference_engine
        self.max_attempts = max_attempts

    def critique_failure(self, task_id: str, error_msg: str) -> Dict[str, Any]:
        """Parses error message and generates critique notes."""
        critique = {
            "task_id": task_id,
            "error_summary": error_msg,
            "suggested_fix": f"Re-evaluate input parameters or environment configuration for error: {error_msg}"
        }
        return critique

    async def attempt_repair(
        self,
        task_id: str,
        error_msg: str,
        current_attempt: int,
        execute_fn
    ) -> Dict[str, Any]:
        if current_attempt > self.max_attempts:
            return {
                "repaired": False,
                "reason": f"Exceeded maximum repair attempts ({self.max_attempts})"
            }

        critique = self.critique_failure(task_id, error_msg)
        logger.info(f"[*] [RepairEngine] Attempt {current_attempt}/{self.max_attempts} for task '{task_id}': {critique['suggested_fix']}")

        # Re-execute via callback
        try:
            repair_result = await execute_fn()
            return {
                "repaired": True,
                "attempt": current_attempt,
                "result": repair_result,
                "critique": critique
            }
        except Exception as e:
            return {
                "repaired": False,
                "attempt": current_attempt,
                "error": str(e)
            }
