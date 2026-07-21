import torch
from ai_core.transformer.base_model import BaseModel

class JarvisMoE(BaseModel):
    """Stub interface for Mixture of Experts architecture."""
    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_num_params(self, non_embedding: bool = True) -> int:
        return 0

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        raise NotImplementedError("JarvisMoE planned for Phase 5 research milestone.")

class JarvisReasoner(BaseModel):
    """Stub interface for CoT reasoning model architecture."""
    def __init__(self, config):
        super().__init__()
        self.config = config

    def get_num_params(self, non_embedding: bool = True) -> int:
        return 0

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        raise NotImplementedError("JarvisReasoner planned for Phase 5 research milestone.")
