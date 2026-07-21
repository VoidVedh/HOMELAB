import torch
from ai_core.config.model_config import ModelConfig
from ai_core.transformer.model import JarvisTransformer
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.inference.engine import InferenceEngine

def test_inference_engine():
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
    tokenizer.train("JARVIS AI OS system", vocab_size=200)
    
    engine = InferenceEngine(model=model, tokenizer=tokenizer, device=torch.device("cpu"))
    generated = engine.generate(prompt="JARVIS", max_new_tokens=10, temperature=0.7)
    
    assert isinstance(generated, str)
    assert len(generated) > 0
