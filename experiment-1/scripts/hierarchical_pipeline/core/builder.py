"""Bottom-up hierarchy builder.

Simple, deterministic bottom-up strategy:
- Each discovered `Part` becomes a node (leaf) initially.
- Compute containment ratios between parts and assign the best parent
  for each candidate child when containment exceeds a threshold.
- Compute node levels (leaf=0) by propagating up the tree.

This is intentionally minimal but useful for Phase 1 experiments.
"""

from typing import List, Optional, Dict, Any
from .interfaces import HierarchyBuilder
from .data import ParseGraph, Node, Part
from .spatial import containment_ratio, pairwise_overlap_matrix
import numpy as np


class BottomUpHierarchyBuilder(HierarchyBuilder):
    """Simple bottom-up hierarchy builder.

    Configuration (via `config`):
    - containment_threshold: float (default 0.7) fraction of child inside parent
    - spatial_threshold: float (not used currently)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        params = self.config.get("params") or {}
        self.containment_threshold = params.get("containment_threshold", 0.7)
        self.spatial_threshold = params.get("spatial_threshold", 0.3)

    def build_hierarchy(self, parts: List[Part]) -> ParseGraph:
        if parts is None:
            raise ValueError("parts cannot be None")

        # Determine image size from parts metadata if available
        image_size = (0, 0)
        image_path = ""
        for p in parts:
            if p.metadata and "image_size" in p.metadata:
                image_size = tuple(p.metadata.get("image_size"))
                break
        for p in parts:
            if p.metadata and "image_path" in p.metadata:
                image_path = p.metadata.get("image_path")
                break

        graph = ParseGraph(image_path=image_path, image_size=image_size)

        if not parts:
            return graph

        # Add nodes (use part.id as node id)
        for p in parts:
            node = Node(id=p.id, part=p, level=0)
            graph.add_node(node)

        # Compute pairwise intersections/areas (CPU fallback inside pairwise helper)
        masks = [p.mask.astype(np.uint8) for p in parts]
        areas, inter = pairwise_overlap_matrix(masks)

        n = len(parts)

        # For each potential child, choose best parent (highest containment)
        parent_of = {p.id: None for p in parts}
        # Map ids to parts for quick lookup later (used to break cycles)
        id_to_part = {p.id: p for p in parts}
        for j in range(n):
            best_parent = None
            best_score = 0.0
            for i in range(n):
                if i == j:
                    continue
                # containment of j inside i
                # fallback to compute containment_ratio if inter/areas insufficient
                if areas[j] == 0:
                    continue
                # Use precomputed intersection
                intersection = float(inter[i][j])
                containment = intersection / areas[j] if areas[j] > 0 else 0.0
                if (
                    containment > self.containment_threshold
                    and containment > best_score
                ):
                    best_score = containment
                    best_parent = parts[i].id

            if best_parent:
                parent_of[parts[j].id] = (best_parent, best_score)

        # Detect and break simple mutual parent-child assignments to avoid
        # creating cycles (which would make level propagation diverge).
        # If A->B and B->A are both selected, keep the direction with
        # higher containment score (or larger area as tiebreaker).
        for child_id, pinfo in list(parent_of.items()):
            if pinfo is None:
                continue
            parent_id, score = pinfo
            other = parent_of.get(parent_id)
            if other is None:
                continue
            other_parent, other_score = other
            if other_parent == child_id:
                # mutual assignment detected between child_id and parent_id
                # choose direction to keep
                if score > other_score:
                    # keep parent_id -> child_id, remove parent for parent_id
                    parent_of[parent_id] = None
                elif score < other_score:
                    parent_of[child_id] = None
                else:
                    # tie: use area as tiebreaker
                    a_child = id_to_part[child_id].get_area()
                    a_parent = id_to_part[parent_id].get_area()
                    if a_parent >= a_child:
                        parent_of[parent_id] = None
                    else:
                        parent_of[child_id] = None

        # Add edges for chosen parent-child relations (after cycle-break)
        for child_id, pinfo in parent_of.items():
            if pinfo is None:
                continue
            parent_id, score = pinfo
            graph.add_edge(
                parent_id,
                child_id,
                relation_type="contains",
                confidence=float(score),
                spatial_features={"containment": float(score)},
            )

        # Compute levels: leaf=0, parent level = 1 + max(child.level)
        # First set leaves to level 0 (already), then propagate up.
        # Add a safety iteration cap to prevent infinite loops in pathological cases.
        changed = True
        max_iters = max(100, len(graph.nodes) * 10)
        iters = 0
        while changed and iters < max_iters:
            iters += 1
            changed = False
            for node in list(graph.nodes.values()):
                if node.children:
                    child_levels = [
                        graph.nodes[c].level for c in node.children if c in graph.nodes
                    ]
                    if child_levels:
                        desired = 1 + max(child_levels)
                        if node.level != desired:
                            node.level = desired
                            changed = True

        if iters >= max_iters:
            # Fallback: assign levels by doing a topological-style pass where
            # nodes with no children remain 0 and parents are set to 1+max(children)
            # without further looping to avoid hanging.
            for node in list(graph.nodes.values()):
                if node.children:
                    child_levels = [
                        graph.nodes[c].level for c in node.children if c in graph.nodes
                    ]
                    if child_levels:
                        node.level = 1 + max(child_levels)

        # Choose root: node with no parent and largest area
        root_candidates = [n for n in graph.nodes.values() if n.parent is None]
        if root_candidates:
            # pick one with largest part area
            areas_map = {node.id: node.part.get_area() for node in root_candidates}
            root_id = max(areas_map, key=areas_map.get)
            graph.root = root_id

        return graph
