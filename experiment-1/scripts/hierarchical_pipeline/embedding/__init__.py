"""Embedding generation module for hierarchical visual representations.

This module provides multiple embedding methods for encoding visual parts:
- DummyEmbedding: Random embeddings for testing
- DINOEmbedding: Self-supervised ViT embeddings
- CLIPEmbedding: Vision-language aligned embeddings
- MAEEmbedding: Masked autoencoder embeddings

Also includes hyperbolic projection utilities for hierarchical representation.
"""

from .methods import DummyEmbedding, DINOEmbedding, CLIPEmbedding, MAEEmbedding
from .cache import EmbeddingCache
from .hyperbolic import HyperbolicProjection, HyperbolicDistance, verify_hyperbolic_constraints

__all__ = [
    "DummyEmbedding",
    "DINOEmbedding", 
    "CLIPEmbedding",
    "MAEEmbedding",
    "EmbeddingCache",
    "HyperbolicProjection",
    "HyperbolicDistance",
    "verify_hyperbolic_constraints",
]

