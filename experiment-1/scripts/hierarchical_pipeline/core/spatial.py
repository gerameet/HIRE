"""Spatial relationship utilities (GPU-accelerated).

Provides simple, tested functions to compute overlap, containment and IoU
between binary masks using PyTorch tensors on the specified device. These
utilities are intentionally small and have CPU fallbacks to keep the
dependency surface minimal.
"""

from typing import List, Tuple
import numpy as np

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _to_tensor(mask: np.ndarray, device: str = "cpu"):
    """Convert boolean/uint8 mask to torch.bool tensor on device.

    Falls back to NumPy if PyTorch is unavailable.
    """
    if not HAS_TORCH:
        return mask.astype(bool)

    t = torch.from_numpy(np.ascontiguousarray(mask.astype(np.uint8)))
    t = t.to(dtype=torch.bool)
    if device and device != "cpu":
        try:
            t = t.to(device)
        except Exception:
            # If device invalid, keep on CPU
            pass
    return t


def iou(mask_a: np.ndarray, mask_b: np.ndarray, device: str = "cpu") -> float:
    """Compute Intersection-over-Union between two binary masks.

    Args:
        mask_a, mask_b: 2D binary masks (numpy arrays)
        device: torch device string (e.g., "cuda" or "cpu")

    Returns:
        IoU as float in [0, 1]
    """
    if not HAS_TORCH:
        a = mask_a.astype(bool)
        b = mask_b.astype(bool)
        inter = float((a & b).sum())
        union = float((a | b).sum())
        return 0.0 if union == 0 else inter / union

    ta = _to_tensor(mask_a, device)
    tb = _to_tensor(mask_b, device)
    inter = float((ta & tb).sum().item())
    union = float((ta | tb).sum().item())
    return 0.0 if union == 0 else inter / union


def containment_ratio(
    parent_mask: np.ndarray, child_mask: np.ndarray, device: str = "cpu"
) -> float:
    """Compute fraction of child's area that is contained inside parent.

    Returns intersection_area / child_area (0..1). If child area is zero,
    returns 0.0.
    """
    if not HAS_TORCH:
        p = parent_mask.astype(bool)
        c = child_mask.astype(bool)
        c_area = float(c.sum())
        if c_area == 0:
            return 0.0
        inter = float((p & c).sum())
        return inter / c_area

    tp = _to_tensor(parent_mask, device)
    tc = _to_tensor(child_mask, device)
    c_area = float(tc.sum().item())
    if c_area == 0:
        return 0.0
    inter = float((tp & tc).sum().item())
    return inter / c_area


def pairwise_overlap_matrix(
    masks: List[np.ndarray], device: str = "cpu"
) -> Tuple[List[float], List[List[float]]]:
    """Compute areas and pairwise intersections for a list of masks.

    Returns (areas, intersection_matrix) where areas is list of pixel counts
    and intersection_matrix is NxN matrix of integer intersection counts.
    Uses PyTorch matmul for GPU acceleration when available.
    """
    n = len(masks)
    if n == 0:
        return [], []

    if not HAS_TORCH:
        bool_masks = [m.astype(bool) for m in masks]
        areas = [float(m.sum()) for m in bool_masks]
        inter = [
            [float((bool_masks[i] & bool_masks[j]).sum()) for j in range(n)]
            for i in range(n)
        ]
        return areas, inter

    # Convert to flat uint8 tensor (N, H*W)
    flat = []
    for m in masks:
        t = _to_tensor(m, device).to(dtype=torch.uint8)
        flat.append(t.view(-1).to(dtype=torch.float32))
    mat = torch.stack(flat, dim=0)  # (N, HW)

    # Areas: sum over columns
    areas = mat.sum(dim=1).cpu().numpy().tolist()

    # Intersection counts = mat @ mat.T
    inter = (mat @ mat.t()).cpu().numpy().tolist()

    return areas, inter
