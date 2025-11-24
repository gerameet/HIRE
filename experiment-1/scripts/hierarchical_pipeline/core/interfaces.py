"""Abstract interfaces for hierarchical pipeline components.

This module defines the core interfaces that all implementations must follow,
enabling modular experimentation with different methods.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

from .data import Part, ParseGraph


class PartDiscoveryMethod(ABC):
    """Abstract interface for part discovery methods.

    Implementations include:
    - Existing segmentation models (YOLO, SAM, Mask2Former)
    - Object-centric models (Slot Attention, COCA)
    - Foundation models (SAM with auto-mask generation)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the part discovery method.

        Args:
            config: Configuration dictionary containing method-specific parameters
        """
        self.config = config

    @abstractmethod
    def discover_parts(self, image: np.ndarray) -> List[Part]:
        """Discover parts in an image.

        Args:
            image: Input image as numpy array (H, W, C) with dtype uint8

        Returns:
            List of Part objects containing masks, bboxes, and optional features

        Raises:
            ValueError: If image format is invalid
            RuntimeError: If discovery fails
        """
        pass

    def discover_parts_batch(self, images: List[np.ndarray]) -> List[List[Part]]:
        """Discover parts in a batch of images.

        Default implementation processes images one by one.
        Override for true batch processing with GPU acceleration.

        Args:
            images: List of input images

        Returns:
            List of part lists (one list per image)
        """
        return [self.discover_parts(img) for img in images]

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about this discovery method.

        Returns:
            Dictionary with method metadata
        """
        return {
            "method_class": self.__class__.__name__,
            "config": self.config,
        }


class EmbeddingMethod(ABC):
    """Abstract interface for embedding generation methods.

    Implementations include:
    - Self-supervised encoders (DINO, MAE, MoCo)
    - Vision-language models (CLIP)
    - Hyperbolic projections
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding method.

        Args:
            config: Configuration dictionary containing method-specific parameters
        """
        self.config = config
        self.embedding_dim = config.get("embedding_dim", 768)

    @abstractmethod
    def embed_part(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate embedding for a masked region.

        Args:
            image: Full image as numpy array (H, W, C)
            mask: Binary mask for the part (H, W) with dtype bool or uint8

        Returns:
            Embedding vector as 1D numpy array (normalized)

        Raises:
            ValueError: If image or mask format is invalid
            RuntimeError: If embedding generation fails
        """
        pass

    def embed_batch(
        self, images: List[np.ndarray], masks: List[np.ndarray]
    ) -> np.ndarray:
        """Generate embeddings for a batch of masked regions.

        Default implementation processes parts one by one.
        Override for true batch processing with GPU acceleration.

        Args:
            images: List of full images
            masks: List of binary masks (one per image)

        Returns:
            Embedding matrix as 2D numpy array (N, embedding_dim)
        """
        embeddings = []
        for img, mask in zip(images, masks):
            embeddings.append(self.embed_part(img, mask))
        return np.stack(embeddings, axis=0)

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings produced by this method.

        Returns:
            Embedding dimension
        """
        return self.embedding_dim

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about this embedding method.

        Returns:
            Dictionary with method metadata
        """
        return {
            "method_class": self.__class__.__name__,
            "embedding_dim": self.embedding_dim,
            "config": self.config,
        }


class HierarchyBuilder(ABC):
    """Abstract interface for hierarchy construction methods.

    Implementations include:
    - Bottom-up spatial clustering
    - Top-down grammar parsing
    - Hybrid approaches
    - Graph neural networks
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the hierarchy builder.

        Args:
            config: Configuration dictionary containing method-specific parameters
        """
        self.config = config

    @abstractmethod
    def build_hierarchy(self, parts: List[Part]) -> ParseGraph:
        """Construct hierarchical parse graph from parts.

        Args:
            parts: List of discovered parts with embeddings

        Returns:
            ParseGraph with nodes (parts) and edges (relationships)

        Raises:
            ValueError: If parts list is invalid
            RuntimeError: If hierarchy construction fails
        """
        pass

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about this hierarchy builder.

        Returns:
            Dictionary with method metadata
        """
        return {
            "method_class": self.__class__.__name__,
            "config": self.config,
        }
