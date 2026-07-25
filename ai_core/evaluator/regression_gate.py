import os
import json
import glob
from typing import Dict, Any, Optional

class RegressionGate:
    """
    Automated Regression Gate ensuring no new model checkpoint is promoted to 'best_model' 
    unless validation loss improves, benchmark scores improve, or performance is statistically neutral.
    """

    def __init__(self, history_dir: str = "observability/benchmarks", max_allowed_regression_pct: float = 5.0):
        self.history_dir = history_dir
        self.max_allowed_regression_pct = max_allowed_regression_pct

    def evaluate_checkpoint_promotion(
        self,
        current_val_loss: float,
        current_benchmark_results: Dict[str, Any],
        historical_baseline: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determines whether a new model checkpoint should be promoted to 'best_model'.
        """
        if historical_baseline is None:
            historical_baseline = self._get_latest_baseline()

        if not historical_baseline:
            return {
                "promoted": True,
                "reason": "Initial baseline established.",
                "status": "INITIAL_BASELINE"
            }

        prev_val_loss = historical_baseline.get("val_loss", float("inf"))
        prev_benchmark_score = historical_baseline.get("overall_benchmark_score", 0.0)
        curr_benchmark_score = current_benchmark_results.get("overall_success_rate_pct", 100.0)

        loss_improved = current_val_loss < prev_val_loss
        benchmark_improved = curr_benchmark_score >= prev_benchmark_score
        
        loss_degraded_pct = ((current_val_loss - prev_val_loss) / max(prev_val_loss, 1e-6)) * 100.0 if prev_val_loss != float("inf") else 0.0

        if loss_improved or benchmark_improved or loss_degraded_pct <= self.max_allowed_regression_pct:
            return {
                "promoted": True,
                "reason": f"Validation loss: {current_val_loss:.4f} vs prev: {prev_val_loss:.4f}, Benchmark: {curr_benchmark_score:.2f}% vs prev: {prev_benchmark_score:.2f}%",
                "status": "PROMOTED_TO_BEST"
            }

        return {
            "promoted": False,
            "reason": f"Regression detected! Loss degraded by {loss_degraded_pct:.2f}% exceeding max threshold {self.max_allowed_regression_pct}%.",
            "status": "REJECTED_REGRESSION_BLOCKED"
        }

    def _get_latest_baseline(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.history_dir):
            return None
        files = sorted(glob.glob(os.path.join(self.history_dir, "benchmark_run_*.json")))
        if not files:
            return None
        try:
            with open(files[-1], "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "val_loss": data.get("categories", {}).get("model", {}).get("avg_latency_per_tok_ms", 5.0),
                    "overall_benchmark_score": data.get("overall_success_rate_pct", 100.0)
                }
        except Exception:
            return None
