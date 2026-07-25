import torch
import pytest
from ai_core.inference.paged_kv import BlockAllocator, PagedKVCache

def test_block_allocator():
    allocator = BlockAllocator(num_blocks=4, block_size=16)
    assert allocator.free_block_count == 4

    b0 = allocator.allocate()
    b1 = allocator.allocate()
    assert allocator.free_block_count == 2
    assert b0 == 0 and b1 == 1

    allocator.free(b0)
    assert allocator.free_block_count == 3

def test_paged_kv_cache_append_and_retrieve():
    n_layers = 2
    n_kv_heads = 2
    head_dim = 16
    paged_cache = PagedKVCache(n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim, block_size=4, max_blocks=16)

    seq_id = 1
    paged_cache.init_sequence(seq_id)

    # Append 5 tokens (crosses block boundary at block_size=4)
    for pos in range(5):
        k_tok = torch.full((n_kv_heads, head_dim), float(pos + 1))
        v_tok = torch.full((n_kv_heads, head_dim), float((pos + 1) * 10))
        
        for layer in range(n_layers):
            paged_cache.append(seq_id, layer_idx=layer, k_token=k_tok, v_token=v_tok)

    assert paged_cache.seq_lengths[seq_id] == 5
    assert len(paged_cache.block_tables[seq_id]) == 2  # 5 tokens need 2 blocks of size 4

    k_ret, v_ret = paged_cache.get_kv(seq_id, layer_idx=0)
    assert k_ret.shape == (1, n_kv_heads, 5, head_dim)
    assert v_ret.shape == (1, n_kv_heads, 5, head_dim)
    assert torch.allclose(k_ret[0, 0, 0, :], torch.tensor(1.0))
    assert torch.allclose(k_ret[0, 0, 4, :], torch.tensor(5.0))

    paged_cache.free_sequence(seq_id)
    assert seq_id not in paged_cache.block_tables
