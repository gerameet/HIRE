"""Adapters for integrating existing pipelines with hierarchical pipeline.

This module provides adapters to bridge the existing segmentation pipeline
with the hierarchical visual pipeline, enabling reuse of existing models
and infrastructure.
"""

from .segmentation import SegmentationAdapter, SegmentationDiscoveryAdapter

__all__ = [
    "SegmentationAdapter",
    "SegmentationDiscoveryAdapter",
]
