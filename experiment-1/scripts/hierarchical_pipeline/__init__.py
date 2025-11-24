"""Hierarchical Visual Pipeline for compositional scene understanding."""

__version__ = "0.1.0"

from .core.interfaces import PartDiscoveryMethod, EmbeddingMethod, HierarchyBuilder
from .core.data import Part, Node, Edge, ParseGraph
from .utils.gpu import GPUManager, get_device, handle_oom
from .config import load_config, create_default_config, validate_config, PipelineConfig
from .adapters import SegmentationAdapter, SegmentationDiscoveryAdapter

__all__ = [
    "PartDiscoveryMethod",
    "EmbeddingMethod",
    "HierarchyBuilder",
    "Part",
    "Node",
    "Edge",
    "ParseGraph",
    "GPUManager",
    "get_device",
    "handle_oom",
    "load_config",
    "create_default_config",
    "validate_config",
    "PipelineConfig",
    "SegmentationAdapter",
    "SegmentationDiscoveryAdapter",
]
