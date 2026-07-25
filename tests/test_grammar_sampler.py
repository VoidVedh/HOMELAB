import torch
import pytest
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.sampler.grammar_sampler import JSONGrammar, GrammarConstrainedSampler

def test_json_grammar_prefix_validation():
    grammar = JSONGrammar()
    
    assert grammar.is_valid_partial('{"action":') == True
    assert grammar.is_valid_partial('{"action": "test", "payload": {}}') == True
    assert grammar.is_valid_partial("invalid non json content xyz 123") == False

    assert grammar.is_valid_complete('{"action": "test"}') == True
    assert grammar.is_valid_complete('{"action":') == False

def test_grammar_constrained_sampler():
    tokenizer = BPETokenizer()
    tokenizer.train('{"action": "terminal:exec", "payload": {"command": "echo hi"}}\n', vocab_size=500)

    sampler = GrammarConstrainedSampler(tokenizer=tokenizer, grammar=JSONGrammar())
    
    logits = torch.randn(1, len(tokenizer.vocab))
    valid_tokens = sampler.get_valid_next_tokens('{"action":')
    assert len(valid_tokens) > 0

    next_tok = sampler.sample(logits=logits, current_text='{"action":', temperature=0.0)
    assert isinstance(next_tok, int)
