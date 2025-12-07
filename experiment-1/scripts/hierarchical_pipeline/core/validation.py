"""Input validation utilities for hierarchical pipeline.

Provides validation functions for images, masks, and other data structures
to ensure consistent error handling and early failure detection.
"""

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray


def validate_image(image: np.ndarray, name: str = "image") -> None:
    """Validate image array format.

    Args:
        image: Input image array
        name: Name for error messages

    Raises:
        ValueError: If image format is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be ndarray, got {type(image)}")

    if image.ndim not in [2, 3]:
        raise ValueError(
            f"{name} must be 2D (H,W) or 3D (H,W,C), got shape {image.shape}"
        )

    if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
        raise ValueError(f"{name} must have 1, 3, or 4 channels, got {image.shape[2]}")

    if image.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValueError(f"{name} must be uint8 or float type, got {image.dtype}")

    # Check value ranges
    if image.dtype == np.uint8:
        if image.min() < 0 or image.max() > 255:
            raise ValueError(f"{name} has invalid uint8 values")
    elif image.dtype in [np.float32, np.float64]:
        if image.min() < 0 or image.max() > 1:
            # Could be [0, 255] float, warn but don't fail
            if image.max() > 255:
                raise ValueError(
                    f"{name} has invalid float values (expected [0,1] or [0,255])"
                )


def validate_mask(
    mask: np.ndarray, image_shape: Optional[Tuple[int, int]] = None, name: str = "mask"
) -> None:
    """Validate mask array format.

    Args:
        mask: Binary mask array
        image_shape: Expected shape (H, W), if None shape not checked
        name: Name for error messages

    Raises:
        ValueError: If mask format is invalid
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"{name} must be ndarray, got {type(mask)}")

    if mask.ndim != 2:
        raise ValueError(f"{name} must be 2D (H,W), got shape {mask.shape}")

    if image_shape is not None:
        if mask.shape != image_shape[:2]:
            raise ValueError(
                f"{name} shape {mask.shape} doesn't match image shape {image_shape[:2]}"
            )

    # Check if binary
    unique_values = np.unique(mask)
    if not np.all(np.isin(unique_values, [0, 1])):
        # Could be bool type
        if mask.dtype == bool:
            return
        raise ValueError(
            f"{name} must be binary (0/1 or bool), found values: {unique_values}"
        )


def validate_embedding(
    embedding: np.ndarray, expected_dim: Optional[int] = None, name: str = "embedding"
) -> None:
    """Validate embedding vector format.

    Args:
        embedding: Embedding array
        expected_dim: Expected dimensionality, if None not checked
        name: Name for error messages

    Raises:
        ValueError: If embedding format is invalid
    """
    if not isinstance(embedding, np.ndarray):
        raise ValueError(f"{name} must be ndarray, got {type(embedding)}")

    if embedding.ndim != 1:
        raise ValueError(f"{name} must be 1D vector, got shape {embedding.shape}")

    if expected_dim is not None and embedding.shape[0] != expected_dim:
        raise ValueError(
            f"{name} dimension {embedding.shape[0]} doesn't match expected {expected_dim}"
        )

    if embedding.dtype not in [np.float32, np.float64]:
        raise ValueError(f"{name} must be float type, got {embedding.dtype}")

    # Check for NaN or Inf
    if not np.isfinite(embedding).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def validate_bbox(
    bbox: Tuple[int, int, int, int],
    image_shape: Optional[Tuple[int, int]] = None,
    name: str = "bbox",
) -> None:
    """Validate bounding box format.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        image_shape: Optional image shape for bounds checking
        name: Name for error messages

    Raises:
        ValueError: If bbox format is invalid
    """
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        raise ValueError(f"{name} must be 4-tuple (x1, y1, x2, y2), got {bbox}")

    x1, y1, x2, y2 = bbox

    # Check validity
    if x2 <= x1:
        raise ValueError(f"{name} invalid: x2 ({x2}) must be > x1 ({x1})")

    if y2 <= y1:
        raise ValueError(f"{name} invalid: y2 ({y2}) must be > y1 ({y1})")

    # Check bounds if image shape provided
    if image_shape is not None:
        h, w = image_shape[:2]
        if x1 < 0 or y1 < 0:
            raise ValueError(f"{name} has negative coordinates: ({x1}, {y1})")
        if x2 > w or y2 > h:
            raise ValueError(
                f"{name} ({x1},{y1},{x2},{y2}) exceeds image bounds ({w},{h})"
            )


def validate_config(config: dict, required_keys: list, name: str = "config") -> None:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        name: Name for error messages

    Raises:
        ValueError: If config is invalid
    """
    if not isinstance(config, dict):
        raise ValueError(f"{name} must be dict, got {type(config)}")

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"{name} missing required keys: {missing_keys}")
