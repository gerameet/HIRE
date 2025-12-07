"""Embedding generation methods for visual parts.

Provides multiple embedding strategies:
- DummyEmbedding: Random baseline for testing
- DINOEmbedding: Self-supervised vision transformer
- DINOv2Embedding: Improved self-supervised vision transformer v2
- CLIPEmbedding: Vision-language aligned embeddings
- MAEEmbedding: Masked autoencoder features

Also includes:
- EmbeddingCache: Disk-based caching for fast iteration
"""

from .methods import (
    DummyEmbedding,
    DINOEmbedding,
    DINOv2Embedding,
    CLIPEmbedding,
    MAEEmbedding,
)
from .cache import EmbeddingCache

__all__ = [
    "DummyEmbedding",
    "DINOEmbedding",
    "DINOv2Embedding",
    "CLIPEmbedding",
    "MAEEmbedding",
    "EmbeddingCache",
    "HyperbolicProjection",
    "HyperbolicDistance",
    "verify_hyperbolic_constraints",
]
