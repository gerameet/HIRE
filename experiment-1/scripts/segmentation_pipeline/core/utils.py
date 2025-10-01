from dataclasses import dataclass, field
import os
import zlib
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

from .data import Segment, BoundingBox


def get_device(preferred: Optional[str] = None) -> str:
    """Get the best available device.

    Args:
        preferred: Preferred device ("cuda", "mps", "cpu")

    Returns:
        Device string
    """
    if preferred:
        return preferred

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def mask_to_bbox(mask: np.ndarray) -> BoundingBox:
    """Convert binary mask to bounding box."""
    return BoundingBox.from_mask(mask)


def filter_segments(
    segments: List[Segment],
    min_area: int = 0,
    max_area: Optional[int] = None,
    min_score: float = 0.0,
    labels: Optional[List[str]] = None,
) -> List[Segment]:
    """Filter segments by various criteria.

    Args:
        segments: List of segments to filter
        min_area: Minimum area in pixels
        max_area: Maximum area in pixels (None = no limit)
        min_score: Minimum confidence score
        labels: List of allowed labels (None = all labels)

    Returns:
        Filtered list of segments
    """
    filtered = []
    for seg in segments:
        if seg.area < min_area:
            continue
        if max_area is not None and seg.area > max_area:
            continue
        if seg.score < min_score:
            continue
        if labels is not None and seg.label not in labels:
            continue
        filtered.append(seg)
    return filtered


def compress_mask(mask: np.ndarray) -> bytes:
    """Compress mask to bytes using zlib."""
    mask_uint8 = mask.astype(np.uint8)
    return zlib.compress(mask_uint8.tobytes())


def decompress_mask(compressed: bytes, shape: Tuple[int, int]) -> np.ndarray:
    """Decompress mask from bytes."""
    decompressed = zlib.decompress(compressed)
    return np.frombuffer(decompressed, dtype=np.uint8).reshape(shape)


def normalize_image(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to normalized numpy array."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def ensure_image_pil(image) -> Image.Image:
    """Ensure image is PIL Image."""
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, np.ndarray):
        return Image.fromarray(image)
    else:
        raise TypeError(f"Expected PIL Image or numpy array, got {type(image)}")


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def non_max_suppression(
    segments: List[Segment], iou_threshold: float = 0.5
) -> List[Segment]:
    """Apply Non-Maximum Suppression to segments based on IoU.

    Args:
        segments: List of segments
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered list of segments
    """
    if not segments:
        return []

    # Sort by score (descending)
    sorted_segs = sorted(segments, key=lambda s: s.score, reverse=True)

    keep = []
    while sorted_segs:
        current = sorted_segs.pop(0)
        keep.append(current)

        # Remove overlapping segments
        remaining = []
        for seg in sorted_segs:
            iou = compute_iou(current.mask, seg.mask)
            if iou < iou_threshold:
                remaining.append(seg)
        sorted_segs = remaining

    return keep
