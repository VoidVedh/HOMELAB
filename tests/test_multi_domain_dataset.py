import os
import pytest
from ai_core.tokenizer.bpe_tokenizer import BPETokenizer
from ai_core.datasets.multi_domain_corpus import MultiDomainCorpus, DOMAINS

def test_multi_domain_corpus_manifest_generation(tmp_path):
    tokenizer = BPETokenizer()
    tokenizer.train("JARVIS AI OS system test corpus", vocab_size=256)
    
    out_dir = str(tmp_path / "manifests")
    manifests = MultiDomainCorpus.generate_all_14_domain_manifests(
        tokenizer=tokenizer,
        max_seq_len=32,
        output_dir=out_dir
    )
    
    assert len(manifests) == 14
    for m in manifests:
        assert "id" in m
        assert m["domain"] in DOMAINS
        assert "checksum" in m
        assert m["checksum"].startswith("sha256:")
        assert m["tokens"] > 0
        assert m["documents"] > 0
