"""Tests for labeling system."""

import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

from hierarchical_pipeline.core.data import Part, Node, ParseGraph
from hierarchical_pipeline.labeling.auto_labeler import CLIPAutoLabeler
from hierarchical_pipeline.labeling.propagation import HierarchicalLabelPropagation

class TestLabeling(unittest.TestCase):
    def setUp(self):
        self.parts = [
            Part(id="p1", mask=np.zeros((10,10)), bbox=(0,0,10,10), embedding=np.random.rand(512).astype(np.float32)),
            Part(id="p2", mask=np.zeros((10,10)), bbox=(0,0,5,5), embedding=np.random.rand(512).astype(np.float32))
        ]
        
    def test_auto_labeler(self):
        """Test CLIP auto labeling."""
        mock_embedder = MagicMock()
        mock_embedder.model = MagicMock()
        mock_embedder.processor = MagicMock()
        mock_embedder.device = "cpu"
        
        # Mock text features: (vocab_size=2, dim=512)
        mock_embedder.model.get_text_features.return_value = torch.rand(2, 512)
        
        vocabulary = ["cat", "dog"]
        labeler = CLIPAutoLabeler(mock_embedder, vocabulary)
        
        # Mocked text features are random, so we just check structure of output
        # Manually force text features to have unit norm for deterministic similarity if needed
        # But for unit test, just checking return type is fine
        
        labels = labeler.label_part(self.parts[0], top_k=1)
        self.assertEqual(len(labels), 1)
        self.assertIn(labels[0][0], vocabulary)
        self.assertTrue(0 <= labels[0][1] <= 100) # Scores are scaled by 100

    def test_propagation(self):
        """Test hierarchical label propagation."""
        graph = ParseGraph(image_path="", image_size=(100,100))
        
        # p1 (root) -> p2 (child)
        self.parts[0].labels = ["animal"]
        self.parts[1].labels = ["cat"]
        
        n1 = Node(id="n1", part=self.parts[0], level=1, children=["n2"])
        n2 = Node(id="n2", part=self.parts[1], level=0, parent="n1")
        
        graph.add_node(n1)
        graph.add_node(n2)
        
        prop = HierarchicalLabelPropagation()
        
        # Top down
        prop.propagate_top_down(graph)
        self.assertIn("animal", graph.nodes["n2"].inherited_labels)
        self.assertIn("animal", graph.nodes["n2"].combined_labels)
        
        # Bottom up (reset combined)
        graph.nodes["n1"].combined_labels = []
        prop.propagate_bottom_up(graph)
        
        # n1 should get "cat" from n2
        self.assertIn("cat", graph.nodes["n1"].combined_labels)

if __name__ == "__main__":
    unittest.main()
