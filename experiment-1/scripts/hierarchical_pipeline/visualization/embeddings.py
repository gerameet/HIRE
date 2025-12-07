"""Embedding space visualization tools.

Provides interactive visualizations of high-dimensional embedding spaces
using dimensionality reduction techniques (t-SNE, UMAP, PCA).
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_embedding_space(
    parts: List,
    method: str = "umap",
    color_by: str = "label",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
):
    """Visualize embedding space in 2D using dimensionality reduction.
    
    Args:
        parts: List of Part objects with embeddings
        method: Reduction method ('umap', 'tsne', 'pca')
        color_by: How to color points ('label', 'image', 'level', 'confidence')
        save_path: Optional path to save HTML plot
        title: Optional plot title
        **kwargs: Additional arguments for reduction method
        
    Returns:
        Plotly Figure object
        
    Raises:
        ValueError: If no parts have embeddings or invalid method
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required for embedding visualization. Install with: pip install plotly")
    
    # Filter parts with embeddings
    valid_parts = [
        p for p in parts
        if hasattr(p, "embedding") and p.embedding is not None
    ]
    
    if not valid_parts:
        raise ValueError("No parts have embeddings to visualize")
    
    logger.info(f"Visualizing {len(valid_parts)} embeddings using {method.upper()}")
    
    # Extract embeddings
    embeddings = np.stack([p.embedding for p in valid_parts])
    
    # Reduce to 2D
    coords_2d = _reduce_dimensions(embeddings, method, **kwargs)
    
    # Prepare color data
    color_data, color_label = _prepare_color_data(valid_parts, color_by)
    
    # Prepare hover data
    hover_text = [
        f"ID: {p.id}<br>"
        f"Label: {getattr(p, 'label', 'unknown')}<br>"
        f"Confidence: {getattr(p, 'confidence', 0):.2f}<br>"
        f"Area: {p.get_area() if hasattr(p, 'get_area') else 'N/A'}"
        for p in valid_parts
    ]
    
    # Create scatter plot
    fig = px.scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        color=color_data,
        hover_name=hover_text,
        title=title or f"Embedding Space ({method.upper()})",
        labels={
            "x": f"{method.upper()} Dimension 1",
            "y": f"{method.upper()} Dimension 2",
            "color": color_label
        },
        color_continuous_scale="Viridis" if isinstance(color_data[0], (int, float)) else None
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        hovermode='closest',
        width=1000,
        height=800,
        font=dict(size=12)
    )
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Saved embedding visualization to {save_path}")
    
    return fig


def plot_embedding_clusters(
    parts: List,
    n_clusters: int = 5,
    method: str = "umap",
    save_path: Optional[str] = None,
    **kwargs
):
    """Visualize embedding space with automatic clustering.
    
    Args:
        parts: List of Part objects with embeddings
        n_clusters: Number of clusters for k-means
        method: Reduction method ('umap', 'tsne', 'pca')
        save_path: Optional path to save HTML plot
        **kwargs: Additional arguments for reduction method
        
    Returns:
        Tuple of (Figure, cluster_labels)
    """
    try:
        from sklearn.cluster import KMeans
        import plotly.express as px
    except ImportError:
        raise ImportError("scikit-learn and plotly required. Install with: pip install scikit-learn plotly")
    
    # Filter parts with embeddings
    valid_parts = [
        p for p in parts
        if hasattr(p, "embedding") and p.embedding is not None
    ]
    
    if not valid_parts:
        raise ValueError("No parts have embeddings to visualize")
    
    # Extract embeddings
    embeddings = np.stack([p.embedding for p in valid_parts])
    
    # Cluster in original space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Reduce to 2D for visualization
    coords_2d = _reduce_dimensions(embeddings, method, **kwargs)
    
    # Prepare hover data
    hover_text = [
        f"ID: {p.id}<br>"
        f"Cluster: {cluster_labels[i]}<br>"
        f"Label: {getattr(p, 'label', 'unknown')}<br>"
        f"Confidence: {getattr(p, 'confidence', 0):.2f}"
        for i, p in enumerate(valid_parts)
    ]
    
    # Create scatter plot
    fig = px.scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        color=cluster_labels.astype(str),
        hover_name=hover_text,
        title=f"Embedding Clusters ({n_clusters} clusters, {method.upper()})",
        labels={
            "x": f"{method.upper()} Dimension 1",
            "y": f"{method.upper()} Dimension 2",
            "color": "Cluster"
        }
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        hovermode='closest',
        width=1000,
        height=800
    )
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Saved cluster visualization to {save_path}")
    
    return fig, cluster_labels


def _reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    **kwargs
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D.
    
    Args:
        embeddings: High-dimensional embedding matrix (N, D)
        method: Reduction method ('umap', 'tsne', 'pca')
        n_components: Number of output dimensions (default: 2)
        **kwargs: Method-specific parameters
        
    Returns:
        Low-dimensional coordinates (N, n_components)
    """
    method = method.lower()
    
    if method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn required. Install with: pip install umap-learn")
        
        # Default UMAP parameters
        params = {
            'n_neighbors': kwargs.get('n_neighbors', 15),
            'min_dist': kwargs.get('min_dist', 0.1),
            'metric': kwargs.get('metric', 'cosine'),
            'random_state': kwargs.get('random_state', 42)
        }
        
        reducer = umap.UMAP(n_components=n_components, **params)
        coords = reducer.fit_transform(embeddings)
        
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        # Default t-SNE parameters
        params = {
            'perplexity': min(kwargs.get('perplexity', 30), len(embeddings) - 1),
            'learning_rate': kwargs.get('learning_rate', 200.0),
            'random_state': kwargs.get('random_state', 42),
            'n_iter': kwargs.get('n_iter', 1000)
        }
        
        reducer = TSNE(n_components=n_components, **params)
        coords = reducer.fit_transform(embeddings)
        
    elif method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        reducer = PCA(n_components=n_components, random_state=kwargs.get('random_state', 42))
        coords = reducer.fit_transform(embeddings)
        
        # Log explained variance
        explained_var = reducer.explained_variance_ratio_
        logger.info(f"PCA explained variance: {explained_var.sum():.2%}")
        
    else:
        raise ValueError(f"Unknown reduction method: {method}. Use 'umap', 'tsne', or 'pca'.")
    
    return coords


def _prepare_color_data(parts: List, color_by: str) -> Tuple[List, str]:
    """Prepare color data for visualization.
    
    Args:
        parts: List of Part objects
        color_by: Attribute to color by
        
    Returns:
        Tuple of (color_data, color_label)
    """
    color_by = color_by.lower()
    
    if color_by == "label":
        color_data = [getattr(p, "label", "unknown") for p in parts]
        color_label = "Label"
        
    elif color_by == "image":
        # Extract image filename from metadata
        color_data = [
            Path(p.metadata.get("image_path", "unknown")).stem
            if hasattr(p, "metadata") and "image_path" in p.metadata
            else "unknown"
            for p in parts
        ]
        color_label = "Image"
        
    elif color_by == "level":
        color_data = [getattr(p, "level", 0) for p in parts]
        color_label = "Hierarchy Level"
        
    elif color_by == "confidence":
        color_data = [getattr(p, "confidence", 0.0) for p in parts]
        color_label = "Confidence"
        
    elif color_by == "area":
        color_data = [p.get_area() if hasattr(p, "get_area") else 0 for p in parts]
        color_label = "Area (pixels)"
        
    else:
        # Try to get attribute directly
        color_data = [getattr(p, color_by, "unknown") for p in parts]
        color_label = color_by.capitalize()
    
    return color_data, color_label
