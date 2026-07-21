import os
import pytest
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer

def test_bpe_tokenizer_train_encode_decode(tmp_path):
    text = "JARVIS is a local AI system. JARVIS learns over time."
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=500)
    
    assert len(tokenizer.vocab) >= len(tokenizer.special_tokens)
    
    encoded = tokenizer.encode(text, add_special_tokens=True)
    assert len(encoded) > 0
    assert encoded[0] == tokenizer.special_tokens["<bos>"]
    assert encoded[-1] == tokenizer.special_tokens["<eos>"]
    
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert "JARVIS" in decoded

def test_bpe_tokenizer_save_load(tmp_path):
    text = "Hello world from custom BPE tokenizer."
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=300)
    
    save_path = str(tmp_path / "vocab.json")
    tokenizer.save(save_path)
    assert os.path.exists(save_path)
    
    loaded_tokenizer = BPETokenizer.load(save_path)
    assert len(loaded_tokenizer.vocab) == len(tokenizer.vocab)
