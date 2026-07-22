import torch
from ai_core.config.model_config import ModelConfig
from ai_core.transformer.model import JarvisTransformer
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.evaluator.domain_benchmarks import DomainBenchmarkRunner
from ai_core.evaluator.regression_framework import ModelRegressionFramework

def test_domain_benchmark_runner_and_regression(tmp_path):
    config = ModelConfig(
        vocab_size=200,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=32
    )
    model = JarvisTransformer(config)
    tokenizer = BPETokenizer()
    tokenizer.train("JARVIS AI OS system test dataset", vocab_size=200)

    bench_dir = str(tmp_path / "benchmarks")
    runner = DomainBenchmarkRunner(
        model=model,
        tokenizer=tokenizer,
        benchmark_dir=bench_dir,
        device=torch.device("cpu")
    )
    
    res_1 = runner.run_suite()
    assert "categories" in res_1
    assert "overall_success_rate_pct" in res_1
    assert len(res_1["categories"]) == 7

    res_2 = runner.run_suite()

    regression = ModelRegressionFramework(history_dir=bench_dir)
    comp = regression.compare_with_previous_run(current_metrics=res_2, previous_metrics=res_1)
    
    assert "status" in comp
    assert comp["regressions_detected"] is False
