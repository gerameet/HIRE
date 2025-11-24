"""Visualization utilities for hierarchical parse graphs.

Includes simple NetworkX-based tree plotting and mask overlay utilities
for quick inspection during Phase 1 experimentation.
"""
from typing import Optional, List
import numpy as np


def overlay_masks(image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.5):
    """Overlay binary masks on an image and return an RGB numpy image.

    The function uses simple color cycling and blends masks over the input.
    Requires `matplotlib` for color maps only; falls back to a basic palette.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
    except Exception:
        plt = None

    # Ensure image is HxWx3
    h, w = image.shape[:2]
    if image.ndim == 2:
        img_rgb = np.stack([image] * 3, axis=-1)
    else:
        img_rgb = image.copy()

    # Normalize to float [0,1]
    img = img_rgb.astype(np.float32) / 255.0

    # Prepare colors
    if plt is not None:
        cmap = matplotlib.cm.get_cmap("tab20")
        colors = [cmap(i)[:3] for i in range(20)]
    else:
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

    out = img.copy()
    for i, m in enumerate(masks):
        color = colors[i % len(colors)]
        mask_bool = (m > 0).astype(np.bool_)
        for c in range(3):
            out[..., c] = np.where(mask_bool, out[..., c] * (1 - alpha) + alpha * color[c], out[..., c])

    # Convert back to uint8
    out_img = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return out_img


def plot_parse_tree(parse_graph, image: Optional[np.ndarray] = None, show_overlay: bool = True, ax=None):
    """Plot the parse graph using NetworkX and optionally show mask overlay.

    Args:
        parse_graph: `ParseGraph` instance
        image: Optional image to overlay masks on
        show_overlay: If True and `image` provided, return overlay image
        ax: Optional matplotlib axis for graph plotting

    Returns:
        If `image` is provided and `show_overlay` True, returns overlay image (H,W,3)
        Otherwise returns None
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("plot_parse_tree requires networkx and matplotlib: ") from e

    G = parse_graph.to_networkx()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    pos = None
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G)

    # Node labels: id (and level)
    labels = {n: f"{n}\nL={G.nodes[n].get('level', '?')}" for n in G.nodes}

    nx.draw(G, pos=pos, ax=ax, with_labels=False, node_size=500, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)

    if image is not None and show_overlay:
        # Collect masks in node order
        masks = []
        for node_id in G.nodes:
            pid = G.nodes[node_id].get("part_id")
            # We cannot access the Part objects here unless user supplied them in metadata
            # If the parse_graph nodes contain parts, use them; otherwise skip
            node = parse_graph.nodes.get(node_id)
            if node and node.part is not None:
                masks.append(node.part.mask)

        if masks:
            overlay = overlay_masks(image, masks)
            return overlay

    return None
