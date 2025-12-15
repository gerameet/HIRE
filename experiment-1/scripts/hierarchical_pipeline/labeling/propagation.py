"""Hierarchical label propagation."""

from typing import List, Dict, Set
from collections import Counter

from ..core.data import ParseGraph, Node

class HierarchicalLabelPropagation:
    """Propagate labels through the hierarchy."""
    
    @staticmethod
    def propagate_top_down(graph: ParseGraph):
        """Pass parent labels to children (Inheritance).
        
        Assumes hierarchical structure (parents are more abstract/general).
        """
        # Top-down order: higher levels first
        sorted_nodes = sorted(graph.nodes.values(), key=lambda n: n.level, reverse=True)
        
        for node in sorted_nodes:
            # Gather parent labels
            if node.parent and node.parent in graph.nodes:
                parent_node = graph.nodes[node.parent]
                
                # Inherit from parent's own labels + parent's inherited
                # For now just use parent.part.labels for simplicity
                inherited = parent_node.part.labels + parent_node.inherited_labels
                
                # Filter duplicates?
                node.inherited_labels = list(dict.fromkeys(inherited))
                
            # Combine
            combined = node.part.labels + node.inherited_labels
            node.combined_labels = list(dict.fromkeys(combined))

    @staticmethod
    def propagate_bottom_up(graph: ParseGraph):
        """Aggregate child labels to parents (voting/summary).
        
        Leaf labels vote for parent labels.
        """
        # Bottom-up: lower levels first
        sorted_nodes = sorted(graph.nodes.values(), key=lambda n: n.level)
        
        for node in sorted_nodes:
            if not node.children:
                continue
                
            # Gather child labels
            child_labels = []
            for child_id in node.children:
                if child_id in graph.nodes:
                    child = graph.nodes[child_id]
                    # Use child's primary labels
                    if child.part.labels:
                        child_labels.extend(child.part.labels)
                        
            if not child_labels:
                continue
                
            # Key: Aggregate logic
            # e.g., Majority vote, or just Union
            # Let's say we want top-3 most common child labels to be added to parent
            counts = Counter(child_labels)
            most_common = [l for l, c in counts.most_common(3)]
            
            # Store where? Part labels are intrinsic.
            # Maybe add to 'combined_labels' or metadata?
            # Or assume parent labels should reflect children
            # Let's append to combined_labels if not present
            current = set(node.combined_labels)
            for l in most_common:
                if l not in current:
                    node.combined_labels.append(l)
