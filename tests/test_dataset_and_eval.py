import pytest
from datasets.pipeline import DatasetPipeline
from observability.experiment_tracker import ExperimentTracker

def test_dataset_pipeline(tmp_path):
    pipeline = DatasetPipeline(base_dir=str(tmp_path))
    manifest = pipeline.process_and_create_manifest(
        dataset_name="test_corpus",
        raw_text="JARVIS is an autonomous AI OS.\nJARVIS is an autonomous AI OS.\nDistinct line.",
        license_info="MIT"
    )
    assert manifest["dataset_name"] == "test_corpus"
    assert manifest["splits"]["train_lines"] > 0

def test_experiment_tracker(tmp_path):
    tracker = ExperimentTracker(experiment_dir=str(tmp_path))
    exp_path = tracker.log_experiment(
        exp_id="exp_test_001",
        config_name="config/jarvis0.yaml",
        dataset_manifest={"tokens": 500},
        checkpoint_path="checkpoints/jarvis0_best.pt",
        metrics={"loss": 4.5}
    )
    assert "exp_test_001.json" in exp_path
