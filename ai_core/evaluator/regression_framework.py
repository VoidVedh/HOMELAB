import os
import json
import glob
from typing import Dict, Any, List, Optional

class ModelRegressionFramework:
    """Automated Model Regression Framework comparing current checkpoint metrics against prior versions."""

    def __init__(self, history_dir: str = "observability/benchmarks", max_regression_pct: float = 5.0):
        self.history_dir = history_dir
        self.max_regression_pct = max_regression_pct

    def load_historical_benchmark_runs(self) -> List[Dict[str, Any]]:
        """Loads past benchmark run logs sorted chronologically."""
        if not os.path.exists(self.history_dir):
            return []
        
        pattern = os.path.join(self.history_dir, "benchmark_run_*.json")
        files = sorted(glob.glob(pattern))
        runs = []
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    runs.append(json.load(f))
            except Exception:
                continue
        return runs

    def compare_with_previous_run(
        self,
        current_metrics: Dict[str, Any],
        previous_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compares current benchmark results against previous run and flags metric regressions."""
        if previous_metrics is None:
            past_runs = self.load_historical_benchmark_runs()
            if len(past_runs) >= 2:
                previous_metrics = past_runs[-2]
            elif len(past_runs) == 1 and past_runs[0] != current_metrics:
                previous_metrics = past_runs[0]

        comparison: Dict[str, Any] = {
            "has_previous": previous_metrics is not None,
            "regressions_detected": False,
            "warnings": [],
            "metrics": {}
        }

        if not previous_metrics:
            comparison["status"] = "INITIAL_BASELINE"
            return comparison

        # Helper for metric evaluation
        # higher_is_better: True for tok/sec or success rate; False for latency, loss, perplexity, unk_rate
        def check_metric(name: str, curr_val: float, prev_val: float, higher_is_better: bool):
            if prev_val == 0:
                change_pct = 0.0
            else:
                change_pct = ((curr_val - prev_val) / abs(prev_val)) * 100.0

            is_regression = (change_pct < -self.max_regression_pct) if higher_is_better else (change_pct > self.max_regression_pct)
            
            detail = {
                "current": round(curr_val, 4),
                "previous": round(prev_val, 4),
                "change_pct": round(change_pct, 2),
                "higher_is_better": higher_is_better,
                "is_regression": is_regression
            }
            
            if is_regression:
                comparison["regressions_detected"] = True
                comparison["warnings"].append(
                    f"REGRESSION DETECTED in '{name}': {prev_val} -> {curr_val} ({change_pct:+.2f}%)"
                )
            return detail

        curr_cats = current_metrics.get("categories", {})
        prev_cats = previous_metrics.get("categories", {})

        # 1. Overall Success Rate
        curr_succ = current_metrics.get("overall_success_rate_pct", 0.0)
        prev_succ = previous_metrics.get("overall_success_rate_pct", 0.0)
        comparison["metrics"]["overall_success_rate_pct"] = check_metric(
            "overall_success_rate_pct", curr_succ, prev_succ, higher_is_better=True
        )

        # 2. Model Latency & Tok/Sec
        curr_model = curr_cats.get("model", {})
        prev_model = prev_cats.get("model", {})
        if curr_model and prev_model:
            curr_tok_sec = curr_model.get("tokens_per_sec", 0.0)
            prev_tok_sec = prev_model.get("tokens_per_sec", 0.0)
            comparison["metrics"]["tokens_per_sec"] = check_metric(
                "tokens_per_sec", curr_tok_sec, prev_tok_sec, higher_is_better=True
            )

            curr_lat = curr_model.get("first_token_latency_ms", 0.0)
            prev_lat = prev_model.get("first_token_latency_ms", 0.0)
            comparison["metrics"]["first_token_latency_ms"] = check_metric(
                "first_token_latency_ms", curr_lat, prev_lat, higher_is_better=False
            )

        comparison["status"] = "PASSED" if not comparison["regressions_detected"] else "REGRESSION_FLAGGED"
        return comparison
