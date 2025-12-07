"""Core components for hierarchical visual pipeline."""

from .interfaces import PartDiscoveryMethod, EmbeddingMethod, HierarchyBuilder
from .data import Part, Node, Edge, ParseGraph
from .validation import (
    validate_image,
    validate_mask,
    validate_embedding,
    validate_bbox,
    validate_config,
)
from .exceptions import (
    PipelineError,
    ModelNotFoundError,
    InvalidInputError,
    EmbeddingError,
    SegmentationError,
    HierarchyBuildError,
    EvaluationError,
    ConfigurationError,
    CacheError,
    ValidationError,
)

__all__ = [
    "PartDiscoveryMethod",
    "EmbeddingMethod",
    "HierarchyBuilder",
    "Part",
    "Node",
    "Edge",
    "ParseGraph",
    # Validation
    "validate_image",
    "validate_mask",
    "validate_embedding",
    "validate_bbox",
    "validate_config",
    # Exceptions
    "PipelineError",
    "ModelNotFoundError",
    "InvalidInputError",
    "EmbeddingError",
    "SegmentationError",
    "HierarchyBuildError",
    "EvaluationError",
    "ConfigurationError",
    "CacheError",
    "ValidationError",
]
