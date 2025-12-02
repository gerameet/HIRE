#!/usr/bin/env python3
"""Simple demo of retrieval engine functionality.

Tests the part retrieval system with generated embeddings.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from hierarchical_pipeline.embedding import DummyEmbedding
from hierarchical_pipeline.evaluation.retrieval import PartRetrievalEngine, precision_at_k, recall_at_k
import numpy as np

print("=" * 70)
print("Part Retrieval Engine Demo")
print("=" * 70)

# Create dummy parts with embeddings
print("\n[1/4] Creating test parts with embeddings...")
embedding_method = DummyEmbedding({"embedding_dim": 768, "seed": 42})

class DummyPart:
    def __init__(self, id, embedding):
        self.id = id
        self.embedding = embedding

# Generate 20 parts
parts = []
for i in range(20):
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    emb = embedding_method.embed_part(img, mask)
    part = DummyPart(f"part_{i}", emb)
    parts.append(part)

print(f"  ✓ Created {len(parts)} parts")

# Build retrieval index
print("\n[2/4] Building retrieval index...")
engine = PartRetrievalEngine(embedding_dim=768, use_faiss=False)  # Use brute-force for demo
engine.add_parts(parts)

stats = engine.get_stats()
print(f"  ✓ Index built: {stats['num_parts']} parts, dim={stats['embedding_dim']}")
print(f"  ✓ Using: {'FAISS' if stats['uses_faiss'] else 'Brute-force'}")

# Query for similar parts
print("\n[3/4] Querying for similar parts...")
query_part = parts[0]
results = engine.query(query_part, top_k=5)

print(f"  Query: {query_part.id}")
print(f"  Top-5 Results:")
for rank, (part, distance) in enumerate(results, 1):
    print(f"    {rank}. {part.id} (distance: {distance:.4f})")

# Test metrics
print("\n[4/4] Testing evaluation metrics...")
# Simulate ground truth (parts 0,1,2 are relevant)
ground_truth = parts[:3]

prec = precision_at_k(results, ground_truth, k=5)
rec = recall_at_k(results, ground_truth, k=5)

print(f"  Precision@5: {prec:.3f}")
print(f"  Recall@5: {rec:.3f}")

print("\n" + "=" * 70)
print("✓ Retrieval engine demo complete!")
print("=" * 70)
print("\nNext steps:")
print("  1. Run with real embeddings (DINO/CLIP)")
print("  2. Test on larger dataset (50+ images)")
print("  3. Benchmark retrieval speed")
