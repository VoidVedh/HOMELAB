import os
from typing import Dict, Type, Optional
from ai_core.config.model_config import ModelConfig
from ai_core.transformer.base_model import BaseModel
from ai_core.transformer.model import JarvisTransformer

class ModelRegistry:
    """Registry pattern for dynamically loading models from configuration files or checkpoints."""
    
    _MODELS: Dict[str, Type[BaseModel]] = {
        "transformer": JarvisTransformer,
    }

    @classmethod
    def register_model_type(cls, name: str, model_cls: Type[BaseModel]):
        cls._MODELS[name] = model_cls

    @classmethod
    def load(
        cls,
        model_name: str,
        config_dir: str = "config",
        checkpoint_path: Optional[str] = None,
        device: str = "cpu"
    ) -> BaseModel:
        yaml_path = os.path.join(config_dir, f"{model_name}.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found for '{model_name}' at {yaml_path}")

        config = ModelConfig.from_yaml(yaml_path)
        model_cls = cls._MODELS.get(config.model_type.lower())
        if not model_cls:
            raise ValueError(f"Unknown model_type '{config.model_type}' in config {yaml_path}")

        model = model_cls(config)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_checkpoint(checkpoint_path)

        return model.to(device)
