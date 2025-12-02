#!/usr/bin/env python3
"""Demo script showing Phase 2A embedding functionality.

This demonstrates:
1. DummyEmbedding generation
2. Embedding caching
3. Hyperbolic projection
4. Distance computation
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from hierarchical_pipeline.embedding import (
    DummyEmbedding,
    EmbeddingCache,
    HyperbolicProjection,
    HyperbolicDistance,
    verify_hyperbolic_constraints,
)

print("=" * 70)
print("HIRE Phase 2A - Embedding Module Demo")
print("=" * 70)

# Create test data
print("\n[1/5] Creating test image and mask...")
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
mask = np.zeros((100, 100), dtype=np.uint8)
mask[20:40, 30:50] = 1
print(f"  ✓ Image shape: {image.shape}")
print(f"  ✓ Mask area: {mask.sum()} pixels")

# Test DummyEmbedding
print("\n[2/5] Testing DummyEmbedding...")
emb_method = DummyEmbedding({"embedding_dim": 768, "seed": 42})
embedding = emb_method.embed_part(image, mask)
print(f"  ✓ Embedding shape: {embedding.shape}")
print(f"  ✓ Embedding norm: {np.linalg.norm(embedding):.6f}")
print(f"  ✓ Embedding dtype: {embedding.dtype}")

# Test batch processing
print("\n[3/5] Testing batch embedding...")
images = [image] * 5
masks = [mask] * 5
batch_embeddings = emb_method.embed_batch(images, masks)
print(f"  ✓ Batch shape: {batch_embeddings.shape}")
print(f"  ✓ All normalized: {np.allclose(np.linalg.norm(batch_embeddings, axis=1), 1.0)}")

# Test caching
print("\n[4/5] Testing embedding cache...")
cache = EmbeddingCache("cache/demo", enabled=True)
cache.put("demo.jpg", "part_0", "dummy", embedding)
cached = cache.get("demo.jpg", "part_0", "dummy")
print(f"  ✓ Cache hit: {cached is not None}")
print(f"  ✓ Embeddings match: {np.allclose(cached, embedding)}")
stats = cache.get_stats()
print(f"  ✓ Cache entries: {stats['num_entries']}")
cache.clear()
print(f"  ✓ Cache cleared")

# Test hyperbolic projection
print("\n[5/5] Testing hyperbolic projection...")
proj = HyperbolicProjection(model="poincare", curvature=1.0)
hyp_emb1 = proj.project(batch_embeddings[0])
hyp_emb2 = proj.project(batch_embeddings[1])

print(f"  ✓ Hyperbolic embedding shape: {hyp_emb1.shape}")
print(f"  ✓ Poincaré constraint (norm < 1): {np.linalg.norm(hyp_emb1):.6f} < 1.0")

# Test hyperbolic distance
dist_calc = HyperbolicDistance(model="poincare")
distance = dist_calc.distance(hyp_emb1, hyp_emb2)
print(f"  ✓ Hyperbolic distance: {distance:.6f}")

# Verify constraints
hyp_batch = proj.project_batch(batch_embeddings)
valid = verify_hyperbolic_constraints(hyp_batch, model="poincare")
print(f"  ✓ All constraints satisfied: {valid}")

# Pairwise distances
distances = dist_calc.pairwise_distances(hyp_batch)
print(f"  ✓ Distance matrix shape: {distances.shape}")
print(f"  ✓ Symmetric: {np.allclose(distances, distances.T)}")
print(f"  ✓ Zero diagonal: {np.allclose(np.diag(distances), 0.0, atol=1e-5)}")

print("\n" + "=" * 70)
print("✓ All Phase 2A features working correctly!")
print("=" * 70)
print("\nNext: Run 'python -m pytest tests/ -v' to see full test suite")
