import torch
from ai_core.config.model_config import ModelConfig
from ai_core.transformer.model import JarvisTransformer
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.datasets.synthetic_dataset import SyntheticTechnicalCorpus
from ai_core.evaluator.benchmark_evaluator import ModelIntelligenceEvaluator

def test_model_intelligence_evaluator():
    config = ModelConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=32
    )
    model = JarvisTransformer(config)
    tokenizer = BPETokenizer()
    tokenizer.train(SyntheticTechnicalCorpus.generate_corpus(multiplier=2), vocab_size=256)

    dataset = SyntheticTechnicalCorpus.create_dataset(tokenizer, max_seq_len=32, multiplier=2)
    evaluator = ModelIntelligenceEvaluator(model=model, tokenizer=tokenizer, device=torch.device("cpu"))

    loss_metrics = evaluator.evaluate_loss_and_perplexity(dataset, batch_size=4)
    assert "validation_loss" in loss_metrics
    assert "perplexity" in loss_metrics
    assert loss_metrics["validation_loss"] > 0.0

    fid_metrics = evaluator.evaluate_tokenizer_fidelity([
        "JARVIS is a local AI system.",
        "def hello_world(): return True"
    ])
    assert fid_metrics["unk_rate_pct"] == 0.0
    assert fid_metrics["roundtrip_fidelity_pct"] == 100.0

    bench_metrics = evaluator.benchmark_kv_cache_speedup(prompt="JARVIS", max_new_tokens=5)
    assert "speedup_multiplier" in bench_metrics
