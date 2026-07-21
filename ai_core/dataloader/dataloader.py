from torch.utils.data import DataLoader
from ai_core.datasets.dataset import TextSequenceDataset

def create_dataloader(dataset: TextSequenceDataset, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    """Helper to create PyTorch DataLoader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
