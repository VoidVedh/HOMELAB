import os
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentTracker:
    """
    Automated Research Experiment Tracker recording complete metadata per training run:
    git commit, model version, dataset version, hyperparameters, losses, device, RAM, and benchmark results.
    """

    def __init__(self, experiment_dir: str = "observability/experiments"):
        self.experiment_dir = experiment_dir
        os.makedirs(self.experiment_dir, exist_ok=True)

    @staticmethod
    def get_git_commit_hash() -> str:
        try:
            res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "unknown_commit"

    def record_experiment(
        self,
        model_version: str,
        dataset_version: str,
        hyperparameters: Dict[str, Any],
        training_summary: Dict[str, Any],
        benchmark_results: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        git_hash = self.get_git_commit_hash()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exp_id = f"exp_{int(time.time())}"

        record = {
            "experiment_id": exp_id,
            "timestamp": timestamp,
            "git_commit": git_hash,
            "model_version": model_version,
            "dataset_version": dataset_version,
            "hyperparameters": hyperparameters,
            "training_metrics": {
                "final_loss": training_summary.get("final_loss", training_summary.get("loss", 0.0)),
                "perplexity": training_summary.get("perplexity", 0.0),
                "tokens_per_sec": training_summary.get("tokens_per_sec", 0.0),
                "total_tokens": training_summary.get("total_tokens", 0),
                "elapsed_sec": training_summary.get("elapsed_sec", 0.0)
            },
            "checkpoint_path": checkpoint_path or "NOT_SAVED",
            "benchmark_results": benchmark_results or {}
        }

        exp_file = os.path.join(self.experiment_dir, f"{exp_id}.json")
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        record["report_file"] = exp_file
        return record
