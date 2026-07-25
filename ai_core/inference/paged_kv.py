import torch
from typing import Dict, List, Tuple, Optional

class BlockAllocator:
    """Manages a physical pool of fixed-size memory blocks for key-value caches."""

    def __init__(self, num_blocks: int = 128, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_blocks))
        self.allocated_blocks: set = set()

    def allocate(self) -> int:
        if not self.free_blocks:
            raise MemoryError("No free memory blocks available in BlockAllocator")
        block_id = self.free_blocks.pop(0)
        self.allocated_blocks.add(block_id)
        return block_id

    def free(self, block_id: int):
        if block_id in self.allocated_blocks:
            self.allocated_blocks.remove(block_id)
            self.free_blocks.append(block_id)

    @property
    def free_block_count(self) -> int:
        return len(self.free_blocks)


class PagedKVCache:
    """Block-paged Key-Value cache memory manager avoiding memory fragmentation."""

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_blocks: int = 256,
        device: Optional[torch.device] = None
    ):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = device or torch.device("cpu")

        self.allocator = BlockAllocator(num_blocks=max_blocks, block_size=block_size)
        
        # Physical Block Memory Pools per layer: (max_blocks, n_kv_heads, block_size, head_dim)
        self.k_blocks = [
            torch.zeros(max_blocks, n_kv_heads, block_size, head_dim, device=self.device)
            for _ in range(n_layers)
        ]
        self.v_blocks = [
            torch.zeros(max_blocks, n_kv_heads, block_size, head_dim, device=self.device)
            for _ in range(n_layers)
        ]

        # Sequence Block Tables: seq_id -> list of block_ids
        self.block_tables: Dict[int, List[int]] = {}
        self.seq_lengths: Dict[int, int] = {}

    def init_sequence(self, seq_id: int):
        self.block_tables[seq_id] = [self.allocator.allocate()]
        self.seq_lengths[seq_id] = 0

    def append(self, seq_id: int, layer_idx: int, k_token: torch.Tensor, v_token: torch.Tensor):
        """
        Appends a single token key/value tensor to sequence cache.
        k_token, v_token shape: (n_kv_heads, head_dim)
        """
        if seq_id not in self.block_tables:
            self.init_sequence(seq_id)

        seq_len = self.seq_lengths[seq_id]
        table = self.block_tables[seq_id]

        block_idx = seq_len // self.block_size
        offset = seq_len % self.block_size

        if block_idx >= len(table):
            table.append(self.allocator.allocate())

        physical_block_id = table[block_idx]

        self.k_blocks[layer_idx][physical_block_id, :, offset, :] = k_token
        self.v_blocks[layer_idx][physical_block_id, :, offset, :] = v_token

        if layer_idx == self.n_layers - 1:
            self.seq_lengths[seq_id] = seq_len + 1

    def get_kv(self, seq_id: int, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns full concatenated Key and Value tensors for specified sequence and layer.
        Returns shapes: (1, n_kv_heads, seq_len, head_dim)
        """
        if seq_id not in self.block_tables:
            empty = torch.empty(1, self.n_kv_heads, 0, self.head_dim, device=self.device)
            return empty, empty

        seq_len = self.seq_lengths[seq_id]
        if seq_len == 0:
            empty = torch.empty(1, self.n_kv_heads, 0, self.head_dim, device=self.device)
            return empty, empty

        table = self.block_tables[seq_id]
        k_parts, v_parts = [], []

        for i, physical_block_id in enumerate(table):
            valid_tokens = min(self.block_size, seq_len - i * self.block_size)
            if valid_tokens <= 0:
                break
            k_parts.append(self.k_blocks[layer_idx][physical_block_id, :, :valid_tokens, :])
            v_parts.append(self.v_blocks[layer_idx][physical_block_id, :, :valid_tokens, :])

        k_concat = torch.cat(k_parts, dim=1).unsqueeze(0)  # (1, n_kv_heads, seq_len, head_dim)
        v_concat = torch.cat(v_parts, dim=1).unsqueeze(0)

        return k_concat, v_concat

    def free_sequence(self, seq_id: int):
        if seq_id in self.block_tables:
            for block_id in self.block_tables[seq_id]:
                self.allocator.free(block_id)
            del self.block_tables[seq_id]
            del self.seq_lengths[seq_id]
