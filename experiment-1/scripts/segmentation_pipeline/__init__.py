"""
segmentation_pipeline/__init__.py
"""
from .core.base import SegmentationModel
from .core.data import Segment, SegmentationResult
from .models import get_model
from .pipeline import SegmentationPipeline

__version__ = "0.0.1"
__all__ = [
    "SegmentationModel",
    "Segment", 
    "SegmentationResult",
    "get_model",
    "SegmentationPipeline",
]