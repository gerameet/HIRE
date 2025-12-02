"""Unit tests for embedding methods and cache system."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_pipeline.embedding import (
    DummyEmbedding,
    EmbeddingCache,
)

# Test data fixtures
@pytest.fixture
def test_image():
    """Create a test image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def test_mask():
    """Create a test mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 30:50] = 1
    return mask


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    cache_dir = tempfile.mkdtemp()
    yield cache_dir
    shutil.rmtree(cache_dir)


class TestDummyEmbedding:
    """Tests for DummyEmbedding."""
    
    def test_initialization(self):
        """Test DummyEmbedding initialization."""
        emb = DummyEmbedding({"embedding_dim": 512, "seed": 123})
        assert emb.embedding_dim == 512
        assert emb.seed == 123
        assert emb.get_embedding_space() == "euclidean"
    
    def test_embed_part(self, test_image, test_mask):
        """Test single embedding generation."""
        emb = DummyEmbedding({"embedding_dim": 768})
        result = emb.embed_part(test_image, test_mask)
        
        # Check shape
        assert result.shape == (768,)
        
        # Check dtype
        assert result.dtype == np.float32
        
        # Check normalization (should be unit vector)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5
    
    def test_deterministic(self, test_image, test_mask):
        """Test that same seed gives same embeddings."""
        emb1 = DummyEmbedding({"embedding_dim": 768, "seed": 42})
        emb2 = DummyEmbedding({"embedding_dim": 768, "seed": 42})
        
        result1 = emb1.embed_part(test_image, test_mask)
        result2 = emb2.embed_part(test_image, test_mask)
        
        # Should be identical
        np.testing.assert_array_equal(result1, result2)
    
    def test_different_seeds(self, test_image, test_mask):
        """Test that different seeds give different embeddings."""
        emb1 = DummyEmbedding({"embedding_dim": 768, "seed": 42})
        emb2 = DummyEmbedding({"embedding_dim": 768, "seed": 123})
        
        result1 = emb1.embed_part(test_image, test_mask)
        result2 = emb2.embed_part(test_image, test_mask)
        
        # Should be different
        assert not np.allclose(result1, result2)
    
    def test_batch_processing(self, test_image, test_mask):
        """Test batch embedding generation."""
        emb = DummyEmbedding({"embedding_dim": 768})
        
        images = [test_image] * 5
        masks = [test_mask] * 5
        
        results = emb.embed_batch(images, masks)
        
        # Check shape        assert results.shape == (5, 768)
        
        # Check dtype
        assert results.dtype == np.float32
        
        # Check all normalized
        norms = np.linalg.norm(results, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""
    
    def test_initialization(self, temp_cache_dir):
        """Test cache initialization."""
        cache = EmbeddingCache(temp_cache_dir, enabled=True)
        assert cache.enabled
        assert cache.cache_dir.exists()
    
    def test_cache_miss(self, temp_cache_dir):
        """Test cache miss returns None."""
        cache = EmbeddingCache(temp_cache_dir, enabled=True)
        result = cache.get("image.jpg", "part_0", "dino")
        assert result is None
    
    def test_cache_put_get(self, temp_cache_dir):
        """Test putting and getting embeddings."""
        cache = EmbeddingCache(temp_cache_dir, enabled=True)
        
        # Create test embedding
        embedding = np.random.randn(768).astype(np.float32)
        
        # Put in cache
        cache.put("image.jpg", "part_0", "dino", embedding)
        
        # Get from cache
        result = cache.get("image.jpg", "part_0", "dino")
        
        # Should match
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
    
    def test_cache_key_uniqueness(self, temp_cache_dir):
        """Test that different inputs produce different cache keys."""
        cache = EmbeddingCache(temp_cache_dir, enabled=True)
        
        emb1 = np.random.randn(768).astype(np.float32)
        emb2 = np.random.randn(768).astype(np.float32)
        
        # Store with different part IDs
        cache.put("image.jpg", "part_0", "dino", emb1)
        cache.put("image.jpg", "part_1", "dino", emb2)
        
        # Retrieve and check they're different
        result1 = cache.get("image.jpg", "part_0", "dino")
        result2 = cache.get("image.jpg", "part_1", "dino")
        
        np.testing.assert_array_equal(result1, emb1)
        np.testing.assert_array_equal(result2, emb2)
        assert not np.allclose(result1, result2)
    
    def test_cache_config_invalidation(self, temp_cache_dir):
        """Test that different configs create different cache entries."""
        cache = EmbeddingCache(temp_cache_dir, enabled=True)
        
        emb1 = np.random.randn(768).astype(np.float32)
        emb2 = np.random.randn(768).astype(np.float32)
        
        # Store with different configs
        cache.put("image.jpg", "part_0", "dino", emb1, config={"layer": 11})
        cache.put("image.jpg", "part_0", "dino", emb2, config={"layer": 12})
        
        # Retrieve with specific configs
        result1 = cache.get("image.jpg", "part_0", "dino", config={"layer": 11})
        result2 = cache.get("image.jpg", "part_0", "dino", config={"layer": 12})
        
        np.testing.assert_array_equal(result1, emb1)
        np.testing.assert_array_equal(result2, emb2)
    
    def test_cache_disabled(self, temp_cache_dir):
        """Test that disabled cache always returns None."""
        cache = EmbeddingCache(temp_cache_dir, enabled=False)
        
        embedding = np.random.randn(768).astype(np.float32)
        
        # Try to put
        cache.put("image.jpg", "part_0", "dino", embedding)
        
        # Get should return None
        result = cache.get("image.jpg", "part_0", "dino")
        assert result is None
    
    def test_cache_stats(self, temp_cache_dir):
        """Test cache statistics."""
        cache = EmbeddingCache(temp_cache_dir, enabled=True)
        
        # Initially empty
        stats = cache.get_stats()
        assert stats["enabled"]
        assert stats["num_entries"] == 0
        
        # Add some entries
        for i in range(5):
            emb = np.random.randn(768).astype(np.float32)
            cache.put(f"image_{i}.jpg", "part_0", "dino", emb)
        
        # Check stats updated
        stats = cache.get_stats()
        assert stats["num_entries"] == 5
        assert stats["total_size_mb"] > 0
    
    def test_cache_clear(self, temp_cache_dir):
        """Test clearing cache."""
        cache = EmbeddingCache(temp_cache_dir, enabled=True)
        
        # Add entries
        for i in range(3):
            emb = np.random.randn(768).astype(np.float32)
            cache.put(f"image_{i}.jpg", "part_0", "dino", emb)
        
        # Verify entries exist
        assert cache.get_stats()["num_entries"] == 3
        
        # Clear cache
        cache.clear()
        
        # Verify cache is empty
        assert cache.get_stats()["num_entries"] == 0
        
        # Verify gets return None
        for i in range(3):
            result = cache.get(f"image_{i}.jpg", "part_0", "dino")
            assert result is None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
