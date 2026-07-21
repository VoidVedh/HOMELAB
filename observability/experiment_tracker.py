import os
import json
import time
import subprocess
from typing import Dict, Any, Optional

class ExperimentTracker:
    """Logs reproducible experiment runs tracking Git Commit SHA, Config, Manifest, Checkpoints, and Metrics."""

    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)

    def _get_git_commit_sha(self) -> str:
        try:
            res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "unknown_commit"

    def log_experiment(
        self,
        exp_id: str,
        config_name: str,
        dataset_manifest: Optional[Dict[str, Any]],
        checkpoint_path: str,
        metrics: Dict[str, Any],
        notes: str = ""
    ) -> str:
        record = {
            "experiment_id": exp_id,
            "timestamp": time.time(),
            "git_sha": self._get_git_commit_sha(),
            "config_name": config_name,
            "dataset_manifest": dataset_manifest or {},
            "checkpoint_path": checkpoint_path,
            "metrics": metrics,
            "notes": notes
        }

        filepath = os.path.join(self.experiment_dir, f"{exp_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return filepath
