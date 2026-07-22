import torch
from ai_core.config.model_config import ModelConfig
from ai_core.transformer.model import JarvisTransformer
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.inference.engine import InferenceEngine
from ai_core.inference.streamer import TokenStreamer

def test_streaming_kv_cache_inference():
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
    
    received_tokens = []
    def on_token(t_str):
        received_tokens.append(t_str)

    streamer = TokenStreamer(callback=on_token)
    stream_gen = engine.generate_stream(prompt="JARVIS", max_new_tokens=10, streamer=streamer, use_kv_cache=True)
    
    yielded_tokens = list(stream_gen)
    
    assert len(yielded_tokens) > 0
    assert len(received_tokens) == len(yielded_tokens)
    assert engine.last_metrics["first_token_latency_ms"] >= 0.0
    assert engine.last_metrics["avg_latency_per_tok_ms"] >= 0.0
    assert engine.last_metrics["tokens_per_sec"] >= 0.0
