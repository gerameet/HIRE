"""Core data structures for hierarchical visual representations.

This module defines the fundamental data types used throughout the pipeline:
- Part: A discovered visual part with mask, bbox, and embedding
- Node: A node in the parse graph representing a part
- Edge: A hierarchical relationship between nodes
- ParseGraph: The complete hierarchical representation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Iterator
import numpy as np
import json


@dataclass
class Part:
    """A discovered visual part.

    Represents a fine-grained segment of an image discovered through
    object-centric learning, segmentation, or other methods.
    """

    id: str
    mask: np.ndarray  # Binary mask (H, W) with dtype bool or uint8
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    features: Optional[np.ndarray] = None  # Raw features from discovery
    embedding: Optional[np.ndarray] = None  # Semantic embedding
    confidence: float = 1.0  # Discovery confidence
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize data."""
        # Ensure mask is 2D
        if self.mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape {self.mask.shape}")

        # Ensure mask is bool or uint8
        if self.mask.dtype not in [np.bool_, np.uint8]:
            self.mask = (self.mask > 0).astype(np.uint8)
            
        # Validate bbox
        if len(self.bbox) != 4:
            raise ValueError(f"BBox must be (x1, y1, x2, y2), got {self.bbox}")
        
        x1, y1, x2, y2 = self.bbox
        if x1 > x2 or y1 > y2:
            # Auto-fix or raise? Let's raise for now to catch bugs
            raise ValueError(f"Invalid bbox coordinates: {self.bbox}")

    def get_area(self) -> int:
        """Calculate the area of this part in pixels."""
        return int(self.mask.sum())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization).

        Note: mask, features, and embedding are not included in dict form.
        Use separate serialization for those.
        """
        return {
            "id": self.id,
            "bbox": self.bbox,
            "confidence": float(self.confidence),
            "area": self.get_area(),
            "metadata": self.metadata,
        }


@dataclass
class Node:
    """A node in the parse graph.

    Represents a visual concept at a specific level of the hierarchy.
    """

    id: str
    part: Part
    level: int  # Hierarchy level (0=leaf, higher=more abstract)
    concept: Optional[str] = None  # Aligned concept from knowledge graph
    concept_confidence: float = 0.0
    children: List[str] = field(default_factory=list)  # Child node IDs
    parent: Optional[str] = None  # Parent node ID

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return {
            "id": self.id,
            "part_id": self.part.id,
            "level": self.level,
            "concept": self.concept,
            "concept_confidence": float(self.concept_confidence),
            "children": self.children,
            "parent": self.parent,
        }


@dataclass
class Edge:
    """An edge in the parse graph.

    Represents a hierarchical relationship between two nodes.
    """

    parent_id: str
    child_id: str
    relation_type: str  # "contains", "part_of", "adjacent", etc.
    confidence: float = 1.0
    spatial_features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "relation_type": self.relation_type,
            "confidence": float(self.confidence),
            "spatial_features": self.spatial_features,
        }


@dataclass
class ParseGraph:
    """Hierarchical parse graph for an image.

    Represents the complete compositional structure of a visual scene
    as a directed acyclic graph (DAG) of parts and their relationships.
    """

    image_path: str
    image_size: Tuple[int, int]  # (width, height)
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    root: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate graph consistency.
        
        Returns:
            List of error messages (empty if valid).
        """
        errors = []
        
        # Check root
        if self.root and self.root not in self.nodes:
            errors.append(f"Root node {self.root} not found in nodes")
            
        # Check edges
        for edge in self.edges:
            if edge.parent_id not in self.nodes:
                errors.append(f"Edge parent {edge.parent_id} not found")
            if edge.child_id not in self.nodes:
                errors.append(f"Edge child {edge.child_id} not found")
                
        # Check cycles (simple DFS)
        visited = set()
        path = set()
        
        def visit(node_id):
            visited.add(node_id)
            path.add(node_id)
            for child_id in self.nodes[node_id].children:
                if child_id in path:
                    errors.append(f"Cycle detected involving {child_id}")
                if child_id not in visited:
                    visit(child_id)
            path.remove(node_id)
            
        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)
                
        return errors

    def add_node(self, node: Node) -> None:
        """Add a node (part) to the graph.

        Args:
            node: Node to add
        """
        self.nodes[node.id] = node

    def add_edge(
        self,
        parent_id: str,
        child_id: str,
        relation_type: str = "contains",
        confidence: float = 1.0,
        spatial_features: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add hierarchical edge.

        Args:
            parent_id: ID of parent node
            child_id: ID of child node
            relation_type: Type of relationship
            confidence: Confidence in this relationship
            spatial_features: Optional spatial features
        """
        edge = Edge(
            parent_id=parent_id,
            child_id=child_id,
            relation_type=relation_type,
            confidence=confidence,
            spatial_features=spatial_features or {},
        )
        self.edges.append(edge)

        # Update parent-child relationships in nodes
        if parent_id in self.nodes and child_id in self.nodes:
            if child_id not in self.nodes[parent_id].children:
                self.nodes[parent_id].children.append(child_id)
            self.nodes[child_id].parent = parent_id

    def get_level(self, level: int) -> List[Node]:
        """Get all nodes at a specific hierarchy level.

        Args:
            level: Hierarchy level

        Returns:
            List of nodes at that level
        """
        return [node for node in self.nodes.values() if node.level == level]

    def get_children(self, node_id: str) -> List[Node]:
        """Get all children of a node.

        Args:
            node_id: ID of parent node

        Returns:
            List of child nodes
        """
        if node_id not in self.nodes:
            return []
        return [
            self.nodes[cid] for cid in self.nodes[node_id].children if cid in self.nodes
        ]

    def get_parent(self, node_id: str) -> Optional[Node]:
        """Get parent of a node.

        Args:
            node_id: ID of child node

        Returns:
            Parent node or None if no parent
        """
        if node_id not in self.nodes:
            return None
        parent_id = self.nodes[node_id].parent
        return self.nodes.get(parent_id) if parent_id else None

    def traverse_dfs(self, start_node: Optional[str] = None) -> Iterator[Node]:
        """Depth-first traversal of the graph.

        Args:
            start_node: Node ID to start from (defaults to root)

        Yields:
            Nodes in DFS order
        """
        start = start_node or self.root
        if not start or start not in self.nodes:
            return

        visited = set()
        stack = [start]

        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue

            visited.add(node_id)
            node = self.nodes[node_id]
            yield node

            # Add children to stack (in reverse order for left-to-right traversal)
            for child_id in reversed(node.children):
                if child_id not in visited:
                    stack.append(child_id)

    def get_depth(self) -> int:
        """Get maximum depth of the hierarchy.

        Returns:
            Maximum depth (0 if empty)
        """
        if not self.nodes:
            return 0
        return max(node.level for node in self.nodes.values())

    def to_json(self) -> str:
        """Serialize to JSON.

        Returns:
            JSON string representation
        """
        data = {
            "image_path": self.image_path,
            "image_size": self.image_size,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "root": self.root,
            "metadata": self.metadata,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(
        cls, json_str: str, parts_dict: Optional[Dict[str, Part]] = None
    ) -> ParseGraph:
        """Deserialize from JSON.

        Args:
            json_str: JSON string representation
            parts_dict: Optional dictionary mapping part IDs to Part objects
                       (required for full reconstruction)

        Returns:
            ParseGraph instance

        Note:
            Without parts_dict, nodes will have placeholder Part objects.
        """
        data = json.loads(json_str)

        graph = cls(
            image_path=data["image_path"],
            image_size=tuple(data["image_size"]),
            root=data.get("root"),
            metadata=data.get("metadata", {}),
        )

        # Reconstruct nodes
        for node_id, node_data in data["nodes"].items():
            part_id = node_data["part_id"]

            # Get part from dict or create placeholder
            if parts_dict and part_id in parts_dict:
                part = parts_dict[part_id]
            else:
                # Create placeholder part
                part = Part(
                    id=part_id,
                    mask=np.zeros((1, 1), dtype=np.uint8),
                    bbox=(0, 0, 0, 0),
                )

            node = Node(
                id=node_id,
                part=part,
                level=node_data["level"],
                concept=node_data.get("concept"),
                concept_confidence=node_data.get("concept_confidence", 0.0),
                children=node_data.get("children", []),
                parent=node_data.get("parent"),
            )
            graph.add_node(node)

        # Reconstruct edges
        for edge_data in data["edges"]:
            edge = Edge(
                parent_id=edge_data["parent_id"],
                child_id=edge_data["child_id"],
                relation_type=edge_data["relation_type"],
                confidence=edge_data.get("confidence", 1.0),
                spatial_features=edge_data.get("spatial_features", {}),
            )
            graph.edges.append(edge)

        return graph

    def to_networkx(self):
        """Convert to NetworkX graph for analysis.

        Returns:
            NetworkX DiGraph with nodes and edges

        Note:
            Requires networkx to be installed.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for graph conversion. "
                "Install with: pip install networkx"
            )

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            G.add_node(
                node_id,
                level=node.level,
                concept=node.concept,
                concept_confidence=node.concept_confidence,
                part_id=node.part.id,
                bbox=node.part.bbox,
                area=node.part.get_area(),
            )

        # Add edges with attributes
        for edge in self.edges:
            G.add_edge(
                edge.parent_id,
                edge.child_id,
                relation_type=edge.relation_type,
                confidence=edge.confidence,
                **edge.spatial_features,
            )

        # Store graph-level metadata
        G.graph["image_path"] = self.image_path
        G.graph["image_size"] = self.image_size
        G.graph["root"] = self.root
        G.graph.update(self.metadata)

        return G

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the parse graph.

        Returns:
            Dictionary with summary information
        """
        if not self.nodes:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "depth": 0,
                "has_root": False,
            }

        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "depth": self.get_depth(),
            "has_root": self.root is not None,
            "nodes_per_level": {
                level: len(self.get_level(level))
                for level in range(self.get_depth() + 1)
            },
        }
