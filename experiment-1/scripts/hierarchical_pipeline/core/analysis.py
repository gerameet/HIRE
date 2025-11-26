"""Analysis tools for hierarchical parse graphs.

This module implements metrics and analysis utilities for evaluating
the quality and structure of generated hierarchies.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .data import ParseGraph, Node


class HierarchyMetrics:
    """Compute structural and semantic metrics for parse graphs."""

    @staticmethod
    def compute_depth(graph: ParseGraph) -> int:
        """Compute maximum depth of the hierarchy."""
        return graph.get_depth()

    @staticmethod
    def compute_branching_factor(graph: ParseGraph) -> float:
        """Compute average branching factor (children per non-leaf node)."""
        if not graph.nodes:
            return 0.0

        non_leaf_nodes = [n for n in graph.nodes.values() if n.children]

        if not non_leaf_nodes:
            return 0.0

        total_children = sum(len(n.children) for n in non_leaf_nodes)
        return total_children / len(non_leaf_nodes)

    @staticmethod
    def compute_balance(graph: ParseGraph) -> float:
        """Compute tree balance score (0-1).

        Uses a simple metric based on the variation in depth of leaf nodes.
        1.0 means all leaves are at the same depth (perfectly balanced).
        Lower values indicate imbalance.
        """
        if not graph.nodes:
            return 1.0

        leaves = [n for n in graph.nodes.values() if not n.children]

        if not leaves:
            return 1.0

        depths = [n.level for n in leaves]
        if not depths:
            return 1.0

        # Calculate coefficient of variation of leaf depths
        mean_depth = np.mean(depths)
        if mean_depth == 0:
            return 1.0

        std_depth = np.std(depths)
        cv = std_depth / mean_depth

        # Map CV to 0-1 score (heuristic: CV > 1 is very unbalanced)
        return max(0.0, 1.0 - cv)

    @staticmethod
    def compute_coverage(graph: ParseGraph) -> float:
        """Compute pixel coverage of the root node(s)."""
        if not graph.nodes:
            return 0.0

        # If there's a root, use its mask area
        if graph.root and graph.root in graph.nodes:
            root_node = graph.nodes[graph.root]
            # Assuming root mask covers the scene
            # But wait, root might be abstract.
            # Let's check leaf coverage instead.
            pass

        # Calculate union of all leaf masks
        leaves = [n for n in graph.nodes.values() if not n.children]

        if not leaves:
            return 0.0

        # Create a blank mask
        h, w = graph.image_size
        full_mask = np.zeros((h, w), dtype=bool)

        for leaf in leaves:
            if leaf.part and leaf.part.mask is not None:
                # Resize if needed (though parts should match image size)
                if leaf.part.mask.shape != (h, w):
                    # Skip or warn? For now assume correct size
                    continue
                full_mask = full_mask | (leaf.part.mask > 0)

        return float(full_mask.sum()) / (h * w)

    @staticmethod
    def get_all_metrics(graph: ParseGraph) -> Dict[str, float]:
        """Compute all available metrics."""
        return {
            "depth": float(HierarchyMetrics.compute_depth(graph)),
            "branching_factor": HierarchyMetrics.compute_branching_factor(graph),
            "balance": HierarchyMetrics.compute_balance(graph),
            "coverage": HierarchyMetrics.compute_coverage(graph),
            "num_nodes": float(len(graph.nodes)),
            "num_edges": float(len(graph.edges)),
        }
