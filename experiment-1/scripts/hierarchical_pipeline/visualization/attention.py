"""Attention visualization tools.

Provides functionality to visualize attention maps from vision models (DINO, CLIP, etc.)
overlaid on original images.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
import cv2
import logging
import io

logger = logging.getLogger(__name__)


def normalize_attention(attention: np.ndarray) -> np.ndarray:
    """Normalize attention map to 0-1 range.

    Args:
        attention: Raw attention map

    Returns:
        Normalized attention map
    """
    if attention.size == 0:
        return attention

    min_val = attention.min()
    max_val = attention.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(attention)

    return (attention - min_val) / (max_val - min_val)


def apply_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Apply heatmap overlay to image.

    Args:
        image: Original image (H, W, 3)
        heatmap: Attention map (H, W) or (N, N) to be resized
        alpha: Transparency of heatmap (0-1)
        colormap: OpenCV colormap

    Returns:
        Image with heatmap overlay
    """
    # Resize heatmap to match image dimensions
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Normalize heatmap
    heatmap_norm = normalize_attention(heatmap)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert image to RGB if needed (assuming input is RGB)
    # cv2 uses BGR, so if we want to mix, we should be careful.
    # Let's assume input is RGB and we want output RGB.
    # heatmap_color comes out as BGR from cv2.applyColorMap
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return overlay


def plot_attention_grid(
    image: np.ndarray,
    attention_heads: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Multi-head Attention",
) -> plt.Figure:
    """Plot grid of attention heads.

    Args:
        image: Original image (H, W, 3)
        attention_heads: Attention maps from multiple heads (N_heads, H, W)
        output_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    n_heads = attention_heads.shape[0]

    # Determine grid size (approx square)
    cols = int(np.ceil(np.sqrt(n_heads + 1)))  # +1 for original image
    rows = int(np.ceil((n_heads + 1) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot each head
    for i in range(n_heads):
        if i + 1 < len(axes):
            # Overlay heatmap
            heatmap = attention_heads[i]
            overlay = apply_heatmap(image, heatmap, alpha=0.7)

            axes[i + 1].imshow(overlay)
            axes[i + 1].set_title(f"Head {i}")
            axes[i + 1].axis("off")

    # Hide unused subplots
    for i in range(n_heads + 1, len(axes)):
        axes[i].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cls_attention(
    image: np.ndarray,
    attention: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "CLS Token Attention",
) -> plt.Figure:
    """Plot attention of CLS token (usually represents global scene or object).

    Args:
        image: Original image
        attention: Attention map for CLS token
        output_path: Save path
        title: Title

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 5))

    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Attention
    overlay = apply_heatmap(image, attention)
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Attention Overlay")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)

    return fig
