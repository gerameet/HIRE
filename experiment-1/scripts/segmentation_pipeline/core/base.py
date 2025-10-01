from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import numpy as np

from .data import Segment, SegmentationResult


@dataclass
class ModelConfig:
    """Configuration for segmentation models."""

    device: Optional[str] = None
    checkpoint: Optional[str] = None
    model_type: Optional[str] = None
    batch_size: int = 1
    confidence_threshold: float = 0.0
    min_area: int = 0
    max_instances: Optional[int] = None
    use_fp16: bool = False
    extra_params: Dict[str, Any] = field(default_factory=dict)


class SegmentationModel(ABC):
    """Abstract base class for all segmentation models.

    All model implementations should inherit from this class and implement
    the required methods.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model (load weights, setup, etc.).

        This is separate from __init__ to allow lazy loading.
        """
        raise NotImplementedError

    @abstractmethod
    def segment(
        self,
        image: Union[Image.Image, np.ndarray],
        prompts: Optional[Dict[str, Any]] = None,
    ) -> List[Segment]:
        """Run segmentation on a single image.

        Args:
            image: Input image (PIL Image or numpy array)
            prompts: Optional prompts for prompted segmentation models
                    (e.g., text, points, boxes)

        Returns:
            List of Segment objects
        """
        raise NotImplementedError

    def segment_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        prompts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[List[Segment]]:
        """Run segmentation on a batch of images.

        Default implementation processes images one by one.
        Override for true batch processing.

        Args:
            images: List of input images
            prompts: Optional list of prompts (one per image)

        Returns:
            List of segment lists (one list per image)
        """
        results = []
        for i, img in enumerate(images):
            prompt = prompts[i] if prompts and i < len(prompts) else None
            results.append(self.segment(img, prompt))
        return results

    def uses_gpu(self) -> bool:
        """Check if model uses GPU.

        Returns True if the model uses GPU, which affects parallelization strategy.
        """
        return self.config.device and self.config.device.startswith("cuda")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_class": self.__class__.__name__,
            "uses_gpu": self.uses_gpu(),
            "config": self.config.__dict__,
        }

    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
            self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources (optional override)."""
        pass


class PromptableSegmentationModel(SegmentationModel):
    """Base class for models that support prompted segmentation."""

    @abstractmethod
    def segment_with_points(
        self,
        image: Union[Image.Image, np.ndarray],
        points: List[tuple],
        point_labels: Optional[List[int]] = None,
    ) -> List[Segment]:
        """Segment with point prompts."""
        raise NotImplementedError

    @abstractmethod
    def segment_with_boxes(
        self, image: Union[Image.Image, np.ndarray], boxes: List[tuple]
    ) -> List[Segment]:
        """Segment with bounding box prompts."""
        raise NotImplementedError


class TextPromptableSegmentationModel(SegmentationModel):
    """Base class for models that support text-based prompts."""

    @abstractmethod
    def segment_with_text(
        self, image: Union[Image.Image, np.ndarray], text_prompts: Union[str, List[str]]
    ) -> List[Segment]:
        """Segment with text prompts."""
        raise NotImplementedError
