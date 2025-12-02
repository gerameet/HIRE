"""Query-by-example retrieval engine for part similarity search.

Uses FAISS for fast approximate nearest neighbor search on part embeddings.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS not available. Retrieval will use slower brute-force search.")


class PartRetrievalEngine:
    """Fast part retrieval using embedding similarity.

    Supports both FAISS (fast, approximate) and brute-force (slow, exact) search.
    """

    def __init__(self, embedding_dim: int = 768, use_faiss: bool = True):
        """Initialize retrieval engine.

        Args:
            embedding_dim: Dimension of embeddings
            use_faiss: Use FAISS if available (faster)
        """
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss and HAS_FAISS

        self.parts = []
        self.embeddings = []
        self.index = None

        if self.use_faiss:
            logger.info(f"Initializing FAISS index (dim={embedding_dim})")
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            logger.info("Using brute-force search (FAISS not available)")

    def add_parts(self, parts: List[Any]) -> None:
        """Add parts to the search index.

        Args:
            parts: List of Part objects with embeddings
        """
        if not parts:
            return

        # Extract embeddings
        new_embeddings = []
        valid_parts = []

        for p in parts:
            if hasattr(p, "embedding") and p.embedding is not None:
                new_embeddings.append(p.embedding)
                valid_parts.append(p)

        if not new_embeddings:
            logger.warning("No parts with embeddings found")
            return

        # Stack embeddings
        emb_matrix = np.stack(new_embeddings).astype(np.float32)

        # Add to index
        if self.use_faiss:
            self.index.add(emb_matrix)

        # Store parts and embeddings
        self.parts.extend(valid_parts)
        self.embeddings.extend(new_embeddings)

        logger.info(
            f"Added {len(valid_parts)} parts to index (total: {len(self.parts)})"
        )

    def query(self, query_part: Any, top_k: int = 5) -> List[Tuple[Any, float]]:
        """Find K most similar parts.

        Args:
            query_part: Part object with embedding
            top_k: Number of results to return

        Returns:
            List of (part, distance) tuples sorted by similarity
        """
        if not self.parts:
            logger.warning("Index is empty. Add parts first.")
            return []

        if not hasattr(query_part, "embedding") or query_part.embedding is None:
            logger.error("Query part has no embedding")
            return []

        # Get query embedding
        query_emb = query_part.embedding.reshape(1, -1).astype(np.float32)

        if self.use_faiss:
            # FAISS search
            distances, indices = self.index.search(
                query_emb, min(top_k, len(self.parts))
            )
            results = [
                (self.parts[idx], float(dist))
                for dist, idx in zip(distances[0], indices[0])
                if idx < len(self.parts)
            ]
        else:
            # Brute-force search
            distances = []
            for i, emb in enumerate(self.embeddings):
                dist = np.linalg.norm(query_emb[0] - emb)
                distances.append((i, dist))

            # Sort by distance
            distances.sort(key=lambda x: x[1])
            results = [(self.parts[idx], dist) for idx, dist in distances[:top_k]]

        return results

    def query_batch(
        self, query_parts: List[Any], top_k: int = 5
    ) -> List[List[Tuple[Any, float]]]:
        """Batch query for multiple parts.

        Args:
            query_parts: List of Part objects
            top_k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        return [self.query(p, top_k) for p in query_parts]

    def clear(self) -> None:
        """Clear the index."""
        self.parts = []
        self.embeddings = []

        if self.use_faiss:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        logger.info("Index cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary with index stats
        """
        return {
            "num_parts": len(self.parts),
            "embedding_dim": self.embedding_dim,
            "uses_faiss": self.use_faiss,
        }


def precision_at_k(
    results: List[Tuple[Any, float]], ground_truth: List[Any], k: int = 5
) -> float:
    """Compute Precision@K.

    Args:
        results: List of (part, distance) from retrieval
        ground_truth: List of relevant parts
        k: Cutoff for precision

    Returns:
        Precision@K score (0-1)
    """
    if not results or not ground_truth:
        return 0.0

    # Get top-K results
    top_k_parts = [part for part, _ in results[:k]]

    # Count relevant items in top-K
    gt_ids = {id(p) for p in ground_truth}
    relevant_count = sum(1 for p in top_k_parts if id(p) in gt_ids)

    return relevant_count / k


def recall_at_k(
    results: List[Tuple[Any, float]], ground_truth: List[Any], k: int = 5
) -> float:
    """Compute Recall@K.

    Args:
        results: List of (part, distance) from retrieval
        ground_truth: List of relevant parts
        k: Cutoff for recall

    Returns:
        Recall@K score (0-1)
    """
    if not ground_truth:
        return 0.0

    # Get top-K results
    top_k_parts = [part for part, _ in results[:k]]

    # Count relevant items found
    gt_ids = {id(p) for p in ground_truth}
    relevant_count = sum(1 for p in top_k_parts if id(p) in gt_ids)

    return relevant_count / len(ground_truth)


def mean_average_precision(
    all_results: List[List[Tuple[Any, float]]], all_ground_truth: List[List[Any]]
) -> float:
    """Compute Mean Average Precision (mAP).

    Args:
        all_results: List of result lists (one per query)
        all_ground_truth: List of ground truth lists (one per query)

    Returns:
        mAP score (0-1)
    """
    if not all_results or not all_ground_truth:
        return 0.0

    average_precisions = []

    for results, gt in zip(all_results, all_ground_truth):
        if not gt:
            continue

        gt_ids = {id(p) for p in gt}
        precisions = []
        relevant_count = 0

        for rank, (part, _) in enumerate(results, 1):
            if id(part) in gt_ids:
                relevant_count += 1
                precision = relevant_count / rank
                precisions.append(precision)

        if precisions:
            average_precisions.append(sum(precisions) / len(gt))

    return (
        sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    )
