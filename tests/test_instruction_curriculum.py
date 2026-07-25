import pytest
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.datasets.dataset_pipeline import DatasetQualityPipeline
from ai_core.datasets.curriculum import CurriculumV3Pipeline

def test_dataset_quality_pipeline_deduplication_and_scoring():
    pipeline = DatasetQualityPipeline(version="3.0.0_test", min_quality_score=0.4)
    
    raw_samples = [
        {"text": "def add(a, b):\n    return a + b\n", "domain": "Code"},
        {"text": "def add(a, b):\n    return a + b\n", "domain": "Code"},  # Exact Duplicate
        {"text": "short", "domain": "Code"},  # Low quality
        {"text": "<thought>\nStep 1: Check\n</thought>\n<plan>\nPlan 1\n</plan>", "domain": "Reasoning"}
    ]

    result = pipeline.process_raw_samples(raw_samples)
    manifest = result["manifest"]
    
    assert manifest["duplicates_removed"] >= 1
    assert manifest["quality_filtered"] >= 1
    assert manifest["processed_samples"] > 0
    assert len(result["train_split"]) > 0

def test_curriculum_v3_pipeline():
    tokenizer = BPETokenizer()
    tokenizer.train("def add(a, b):\n    return a + b\n", vocab_size=500)
    
    curriculum = CurriculumV3Pipeline(tokenizer=tokenizer, version="3.0.0_test")
    res = curriculum.build_full_curriculum()
    
    assert "manifest" in res
    assert res["manifest"]["processed_samples"] > 0
    samples = res["train_split"] + res["val_split"] + res["test_split"]
    
    stages = set(s["curriculum_stage"] for s in samples)
    assert 1 in stages and 2 in stages and 3 in stages
    assert "sample_id" in samples[0]
    assert "sha256" in samples[0]
    assert "quality_score" in samples[0]
