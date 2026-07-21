import torch
from ai_core.config.model_config import ModelConfig
from ai_core.transformer.model import JarvisTransformer
from ai_core.datasets.dataset import TextSequenceDataset
from ai_core.dataloader.dataloader import create_dataloader
from ai_core.trainer.trainer import Trainer

def test_trainer_single_epoch(tmp_path):
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=16,
        learning_rate=1e-3,
        batch_size=2
    )
    model = JarvisTransformer(config)
    token_ids = list(range(64)) * 5
    dataset = TextSequenceDataset(token_ids, max_seq_len=16)
    loader = create_dataloader(dataset, batch_size=2)
    
    trainer = Trainer(model=model, train_loader=loader, checkpoint_dir=str(tmp_path), device=torch.device("cpu"))
    summary = trainer.train_epoch(epoch=1, max_steps=2)
    
    assert summary["epoch"] == 1
    assert "loss" in summary
    assert summary["loss"] > 0.0
