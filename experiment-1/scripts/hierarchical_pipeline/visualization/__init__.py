"""Visualization package for hierarchical pipeline.

Provides tools for visualizing:
- Embedding spaces (t-SNE, UMAP, PCA)
- Metrics dashboards
- Comparison reports
- Interactive graphs
"""

from .embeddings import plot_embedding_space, plot_embedding_clusters
from .metrics_dashboard import create_metrics_dashboard,generate_comparison_report

__all__ = [
    "plot_embedding_space",
    "plot_embedding_clusters",
    "create_metrics_dashboard",
    "generate_comparison_report",
]
