"""Visualization package for hierarchical pipeline.

Provides tools for visualizing:
- Embedding spaces (t-SNE, UMAP, PCA)
- Metrics dashboards
- Comparison reports
- Interactive graphs
"""

from .embeddings import plot_embedding_space, plot_embedding_clusters
from .metrics_dashboard import create_metrics_dashboard, generate_comparison_report
from .graph import (
    plot_interactive_graph,
    overlay_masks,
    overlay_masks_with_ids,
    plot_parse_tree,
)

__all__ = [
    "plot_embedding_space",
    "plot_embedding_clusters",
    "create_metrics_dashboard",
    "generate_comparison_report",
    "plot_interactive_graph",
    "overlay_masks",
    "overlay_masks_with_ids",
    "plot_parse_tree",
]
