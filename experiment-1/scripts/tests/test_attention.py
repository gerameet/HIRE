"""Tests for attention visualization suite."""

import unittest
import numpy as np
import os
import shutil
from pathlib import Path

from hierarchical_pipeline.embedding.methods import (
    DummyEmbedding,
    DINOEmbedding,
    CLIPEmbedding
)
from hierarchical_pipeline.visualization.attention import (
    apply_heatmap,
    plot_attention_grid
)

class TestAttentionVisualization(unittest.TestCase):
    def setUp(self):
        self.image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.mask = np.zeros((224, 224), dtype=np.uint8)
        self.mask[50:150, 50:150] = 1 # Center square
        self.output_dir = "test_output_attention"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_dummy_attention(self):
        """Test DummyEmbedding returns attention."""
        config = {"seed": 42}
        embedder = DummyEmbedding(config)
        attn = embedder.extract_attention(self.image, self.mask)
        
        self.assertIn("last_attn", attn)
        self.assertIn("cls_attn", attn)
        self.assertEqual(attn["last_attn"].shape, (14, 14))

    def test_heatmap_generation(self):
        """Test heatmap overlay generation."""
        heatmap = np.random.rand(14, 14)
        overlay = apply_heatmap(self.image, heatmap)
        
        self.assertEqual(overlay.shape, self.image.shape)
        self.assertEqual(overlay.dtype, np.uint8)

    def test_grid_plotting(self):
        """Test multi-head grid plotting."""
        heads = np.random.rand(6, 224, 224) # 6 heads
        save_path = os.path.join(self.output_dir, "grid.png")
        fig = plot_attention_grid(self.image, heads, output_path=save_path)
        
        self.assertTrue(os.path.exists(save_path))
        import matplotlib.pyplot as plt
        plt.close(fig)

    # Note: Testing DINO/CLIP requires models which might be slow or fail without GPU/net.
    # We skip them in basic unit tests or mock them if needed.

if __name__ == "__main__":
    unittest.main()
