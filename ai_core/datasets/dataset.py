import torch
from torch.utils.data import Dataset
from typing import List

class TextSequenceDataset(Dataset):
    """PyTorch Dataset producing sliding window token sequences (X) and shifted target tokens (Y)."""
    def __init__(self, token_ids: List[int], max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        # Ensure we have enough tokens for at least one sequence
        if len(token_ids) <= max_seq_len:
            # Pad or repeat if dataset is small
            multiplier = (max_seq_len + 2) // max(len(token_ids), 1) + 1
            token_ids = token_ids * multiplier

        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return max(1, len(self.data) - self.max_seq_len)

    def __getitem__(self, idx: int):
        idx = min(idx, len(self.data) - self.max_seq_len - 1)
        x = self.data[idx : idx + self.max_seq_len]
        y = self.data[idx + 1 : idx + self.max_seq_len + 1]
        return x, y
