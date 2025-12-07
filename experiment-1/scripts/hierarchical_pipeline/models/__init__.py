"""Model management for hierarchical pipeline.

This module provides centralized model downloading, caching, and verification.
"""

from .model_manager import ModelManager, ModelSpec

__all__ = ["ModelManager", "ModelSpec"]
