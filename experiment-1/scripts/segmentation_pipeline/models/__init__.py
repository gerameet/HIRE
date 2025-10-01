from typing import Dict, Type, Optional
from ..core.base import SegmentationModel, ModelConfig


# Model registry
_MODEL_REGISTRY: Dict[str, Type[SegmentationModel]] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls: Type[SegmentationModel]):
        _MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_model(name: str, config: Optional[ModelConfig] = None) -> SegmentationModel:
    """Factory function to create model instances.

    Args:
        name: Model name (e.g., "sam", "mask2former", "yolo")
        config: Model configuration

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model name is not registered
    """
    name_lower = name.lower()
    if name_lower not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")

    model_class = _MODEL_REGISTRY[name_lower]
    config = config or ModelConfig()
    return model_class(config)


def list_available_models() -> list:
    """Get list of available model names."""
    return sorted(_MODEL_REGISTRY.keys())


# Import all model implementations to trigger registration
from . import dummy
from . import sam
from . import mask2former
from . import segformer
from . import yolo
from . import detectron2_models
from . import clipseg


__all__ = [
    "register_model",
    "get_model",
    "list_available_models",
]
