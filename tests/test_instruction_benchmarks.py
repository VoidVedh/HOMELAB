import pytest
from ai_core.config.model_config import ModelConfig
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.transformer.model import JarvisTransformer
from ai_core.evaluator.instruction_benchmarks import InstructionBenchmarkSuite

def test_instruction_benchmark_suite_execution():
    tokenizer = BPETokenizer()
    tokenizer.train("def is_even(n: int) -> bool:\n    return n % 2 == 0\n", vocab_size=500)

    config = ModelConfig(vocab_size=500, d_model=128, n_layers=2, n_heads=4)
    model = JarvisTransformer(config)

    suite = InstructionBenchmarkSuite(model=model, tokenizer=tokenizer)
    results = suite.run_suite()

    assert "humaneval_lite" in results
    assert "mbpp_lite" in results
    assert "cot_reasoning" in results
    assert "needle_in_haystack" in results

    assert results["humaneval_lite"]["pass_at_1"] >= 0.0
    assert results["needle_in_haystack"]["passed"] == True
    assert results["cot_reasoning"]["cot_reasoning_score"] == 100.0
