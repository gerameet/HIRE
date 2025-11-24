"""Test script for core data models.

This script verifies that the Part, Node, Edge, and ParseGraph classes
work correctly, including JSON serialization and NetworkX conversion.
"""

import numpy as np
import sys
from pathlib import Path

# Add hierarchical_pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from hierarchical_pipeline.core.data import Part, Node, Edge, ParseGraph


def test_part_creation():
    """Test Part dataclass creation and validation."""
    print("Testing Part creation...")
    
    # Create a simple part
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:50] = 1
    
    part = Part(
        id="part_1",
        mask=mask,
        bbox=(30, 20, 50, 40),
        confidence=0.95,
        metadata={"source": "test"}
    )
    
    assert part.id == "part_1"
    assert part.get_area() == 400  # 20x20 pixels
    assert part.confidence == 0.95
    assert part.metadata["source"] == "test"
    
    # Test to_dict
    part_dict = part.to_dict()
    assert part_dict["id"] == "part_1"
    assert part_dict["area"] == 400
    assert part_dict["bbox"] == (30, 20, 50, 40)
    
    print("✓ Part creation and validation works")


def test_parse_graph_construction():
    """Test ParseGraph construction with nodes and edges."""
    print("\nTesting ParseGraph construction...")
    
    # Create parts
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[10:90, 10:90] = 1
    part1 = Part(id="part_1", mask=mask1, bbox=(10, 10, 90, 90))
    
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[20:40, 20:40] = 1
    part2 = Part(id="part_2", mask=mask2, bbox=(20, 20, 40, 40))
    
    mask3 = np.zeros((100, 100), dtype=np.uint8)
    mask3[50:70, 50:70] = 1
    part3 = Part(id="part_3", mask=mask3, bbox=(50, 50, 70, 70))
    
    # Create parse graph
    graph = ParseGraph(
        image_path="test_image.jpg",
        image_size=(100, 100),
        root="node_1"
    )
    
    # Add nodes
    node1 = Node(id="node_1", part=part1, level=1, concept="object")
    node2 = Node(id="node_2", part=part2, level=0, concept="part_a")
    node3 = Node(id="node_3", part=part3, level=0, concept="part_b")
    
    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_node(node3)
    
    # Add edges
    graph.add_edge("node_1", "node_2", relation_type="contains", confidence=0.9)
    graph.add_edge("node_1", "node_3", relation_type="contains", confidence=0.85)
    
    # Verify structure
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert graph.root == "node_1"
    assert graph.get_depth() == 1
    
    # Test get_level
    level_0_nodes = graph.get_level(0)
    assert len(level_0_nodes) == 2
    
    level_1_nodes = graph.get_level(1)
    assert len(level_1_nodes) == 1
    
    # Test get_children
    children = graph.get_children("node_1")
    assert len(children) == 2
    assert children[0].id in ["node_2", "node_3"]
    
    # Test get_parent
    parent = graph.get_parent("node_2")
    assert parent is not None
    assert parent.id == "node_1"
    
    # Test DFS traversal
    visited = list(graph.traverse_dfs())
    assert len(visited) == 3
    assert visited[0].id == "node_1"  # Root first
    
    # Test summary
    summary = graph.get_summary()
    assert summary["num_nodes"] == 3
    assert summary["num_edges"] == 2
    assert summary["depth"] == 1
    assert summary["has_root"] is True
    
    print("✓ ParseGraph construction and traversal works")
    
    return graph


def test_json_serialization(graph):
    """Test JSON serialization round-trip."""
    print("\nTesting JSON serialization...")
    
    # Serialize to JSON
    json_str = graph.to_json()
    assert isinstance(json_str, str)
    assert len(json_str) > 0
    
    # Create parts dictionary for deserialization
    parts_dict = {node.part.id: node.part for node in graph.nodes.values()}
    
    # Deserialize from JSON
    reconstructed = ParseGraph.from_json(json_str, parts_dict)
    
    # Verify structure is preserved
    assert reconstructed.image_path == graph.image_path
    assert reconstructed.image_size == graph.image_size
    assert reconstructed.root == graph.root
    assert len(reconstructed.nodes) == len(graph.nodes)
    assert len(reconstructed.edges) == len(graph.edges)
    
    # Verify node properties
    for node_id in graph.nodes:
        assert node_id in reconstructed.nodes
        orig_node = graph.nodes[node_id]
        recon_node = reconstructed.nodes[node_id]
        assert orig_node.level == recon_node.level
        assert orig_node.concept == recon_node.concept
        assert orig_node.concept_confidence == recon_node.concept_confidence
        assert orig_node.children == recon_node.children
        assert orig_node.parent == recon_node.parent
    
    # Verify edge properties
    assert len(reconstructed.edges) == len(graph.edges)
    for orig_edge, recon_edge in zip(graph.edges, reconstructed.edges):
        assert orig_edge.parent_id == recon_edge.parent_id
        assert orig_edge.child_id == recon_edge.child_id
        assert orig_edge.relation_type == recon_edge.relation_type
        assert abs(orig_edge.confidence - recon_edge.confidence) < 1e-6
    
    print("✓ JSON serialization round-trip works")


def test_networkx_conversion(graph):
    """Test NetworkX conversion."""
    print("\nTesting NetworkX conversion...")
    
    try:
        import networkx as nx
    except ImportError:
        print("⚠ NetworkX not installed, skipping test")
        return
    
    # Convert to NetworkX
    G = graph.to_networkx()
    
    # Verify it's a DiGraph
    assert isinstance(G, nx.DiGraph)
    
    # Verify nodes
    assert len(G.nodes) == len(graph.nodes)
    for node_id in graph.nodes:
        assert node_id in G.nodes
        assert G.nodes[node_id]["level"] == graph.nodes[node_id].level
        assert G.nodes[node_id]["concept"] == graph.nodes[node_id].concept
    
    # Verify edges
    assert len(G.edges) == len(graph.edges)
    for edge in graph.edges:
        assert G.has_edge(edge.parent_id, edge.child_id)
        edge_data = G.edges[edge.parent_id, edge.child_id]
        assert edge_data["relation_type"] == edge.relation_type
        assert abs(edge_data["confidence"] - edge.confidence) < 1e-6
    
    # Verify graph metadata
    assert G.graph["image_path"] == graph.image_path
    assert G.graph["image_size"] == graph.image_size
    assert G.graph["root"] == graph.root
    
    print("✓ NetworkX conversion works")


def test_empty_graph():
    """Test empty ParseGraph."""
    print("\nTesting empty ParseGraph...")
    
    graph = ParseGraph(
        image_path="empty.jpg",
        image_size=(100, 100)
    )
    
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0
    assert graph.get_depth() == 0
    
    summary = graph.get_summary()
    assert summary["num_nodes"] == 0
    assert summary["num_edges"] == 0
    assert summary["depth"] == 0
    assert summary["has_root"] is False
    
    # Test JSON serialization of empty graph
    json_str = graph.to_json()
    reconstructed = ParseGraph.from_json(json_str)
    assert len(reconstructed.nodes) == 0
    assert len(reconstructed.edges) == 0
    
    print("✓ Empty ParseGraph works")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Core Data Models")
    print("=" * 60)
    
    try:
        test_part_creation()
        graph = test_parse_graph_construction()
        test_json_serialization(graph)
        test_networkx_conversion(graph)
        test_empty_graph()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
