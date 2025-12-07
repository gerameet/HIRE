"""Visualization utilities for hierarchical parse graphs.

Includes simple NetworkX-based tree plotting and mask overlay utilities
for quick inspection during Phase 1 experimentation.
"""

from typing import Optional, List, Any
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
            out[..., c] = np.where(
                mask_bool, out[..., c] * (1 - alpha) + alpha * color[c], out[..., c]
            )

    # Convert back to uint8
    out_img = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return out_img


def overlay_masks_with_ids(image: np.ndarray, parts: List[Any], alpha: float = 0.5):
    """Overlay binary masks with ID labels on an image.

    Args:
        image: Input image (H, W, 3)
        parts: List of Part objects
        alpha: Transparency for masks

    Returns:
        PIL Image with overlays and text labels
    """
    from PIL import Image, ImageDraw, ImageFont

    # Get base overlay
    masks = [p.mask for p in parts]
    overlay_np = overlay_masks(image, masks, alpha)
    overlay_pil = Image.fromarray(overlay_np)

    draw = ImageDraw.Draw(overlay_pil)

    # Try to load a font, fallback to default
    try:
        # Try a standard font
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    for part in parts:
        if part.mask is None:
            continue

        # Find position for label (center of bbox)
        x1, y1, x2, y2 = part.bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Draw text with outline for visibility
        text = str(part.id)

        # Get text size
        try:
            # Pillow >= 10.0.0
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            w = right - left
            h = bottom - top
        except AttributeError:
            # Older Pillow
            w, h = draw.textsize(text, font=font)

        # Draw background rectangle for text
        draw.rectangle(
            [cx - w // 2 - 2, cy - h // 2 - 2, cx + w // 2 + 2, cy + h // 2 + 2],
            fill=(0, 0, 0, 128),
        )

        # Draw text
        draw.text((cx - w // 2, cy - h // 2), text, fill=(255, 255, 255), font=font)

    return overlay_pil


def plot_parse_tree(
    parse_graph, image: Optional[np.ndarray] = None, show_overlay: bool = True, ax=None
):
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

    overlay = None
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

    # Return both overlay image (or None) and the created figure for saving
    return overlay, fig


def plot_interactive_graph(parse_graph, output_path: str = None):
    """Plot an interactive graph using Plotly.

    Args:
        parse_graph: `ParseGraph` instance
        output_path: Path to save HTML file (optional)

    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
        import networkx as nx
    except ImportError:
        print("Plotly not installed. Skipping interactive plot.")
        return None

    G = parse_graph.to_networkx()

    # Layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Info for hover
        n_data = G.nodes[node]
        info = f"ID: {node}<br>Level: {n_data.get('level')}"
        if n_data.get("concept"):
            info += f"<br>Concept: {n_data.get('concept')}"
        node_text.append(info)

        # Color by level
        node_color.append(n_data.get("level", 0))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            reversescale=True,
            color=node_color,
            size=10,
            colorbar=dict(thickness=15, title="Level", xanchor="left"),
            line_width=2,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Hierarchical Parse Graph", font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    if output_path:
        fig.write_html(output_path)

    return fig
