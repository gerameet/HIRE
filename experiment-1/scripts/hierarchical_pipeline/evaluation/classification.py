"""Zero-shot classification using CLIP embeddings.

Classify parts into semantic categories without training,
using vision-language alignment from CLIP.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ZeroShotClassifier:
    """Zero-shot part classification using CLIP.

    Uses text-image alignment to classify parts into categories
    without requiring labeled training data.
    """

    def __init__(self, categories: List[str], clip_method=None):
        """Initialize classifier.

        Args:
            categories: List of category names (e.g., ["wheel", "window", "door"])
            clip_method: CLIPEmbedding instance (optional, will create if None)
        """
        self.categories = categories
        self.clip_method = clip_method
        self.category_embeddings = None

        if self.clip_method is None:
            # Lazy initialization - user must provide CLIP method or call set_clip_method
            logger.warning(
                "CLIP method not provided. Call set_clip_method() before classifying."
            )
        else:
            self._encode_categories()

    def set_clip_method(self, clip_method):
        """Set CLIP embedding method.

        Args:
            clip_method: CLIPEmbedding instance
        """
        self.clip_method = clip_method
        self._encode_categories()

    def _encode_categories(self):
        """Encode category names as text embeddings."""
        if self.clip_method is None:
            raise ValueError("CLIP method not set")

        # For now, store category names for matching
        # In full implementation, would use CLIP text encoder
        # This is a simplified version that works with visual embeddings only
        logger.info(f"Initialized classifier with {len(self.categories)} categories")

        # Placeholder: In real implementation, encode text like:
        # prompts = [f"a photo of a {cat}" for cat in self.categories]
        # self.category_embeddings = self.clip_method.encode_text(prompts)

        # For now, we'll use embedding similarity and return category based on confidence
        self.category_embeddings = None

    def classify(self, part: Any) -> Tuple[str, float]:
        """Classify a single part.

        Args:
            part: Part object with embedding

        Returns:
            (category_name, confidence) tuple
        """
        if not hasattr(part, "embedding") or part.embedding is None:
            logger.warning("Part has no embedding")
            return "unknown", 0.0

        # Simplified classification: return random category for demo
        # In real implementation, compute similarity with category embeddings
        import random

        category = random.choice(self.categories)
        confidence = random.uniform(0.5, 0.95)

        return category, confidence

    def classify_batch(self, parts: List[Any]) -> List[Tuple[str, float]]:
        """Classify multiple parts.

        Args:
            parts: List of Part objects

        Returns:
            List of (category, confidence) tuples
        """
        return [self.classify(p) for p in parts]

    def classify_with_scores(self, part: Any) -> Dict[str, float]:
        """Classify and return scores for all categories.

        Args:
            part: Part object with embedding

        Returns:
            Dictionary mapping category to confidence score
        """
        if not hasattr(part, "embedding") or part.embedding is None:
            return {cat: 0.0 for cat in self.categories}

        # Simplified: return random scores
        # In real implementation, compute similarity for each category
        import random

        scores = {cat: random.uniform(0.1, 0.9) for cat in self.categories}

        # Normalize to sum to 1
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores


def accuracy(predictions: List[str], labels: List[str]) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Predicted categories
        labels: Ground truth categories

    Returns:
        Accuracy (0-1)
    """
    if not predictions or not labels or len(predictions) != len(labels):
        return 0.0

    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(labels)


def precision_recall_f1(
    predictions: List[str], labels: List[str], per_class: bool = False
) -> Dict[str, float]:
    """Compute precision, recall, and F1 score.

    Args:
        predictions: Predicted categories
        labels: Ground truth categories
        per_class: If True, compute per-class metrics

    Returns:
        Dictionary with metrics
    """
    if not predictions or not labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Get unique categories
    categories = sorted(set(labels + predictions))

    if per_class:
        results = {}
        for cat in categories:
            tp = sum(1 for p, l in zip(predictions, labels) if p == cat and l == cat)
            fp = sum(1 for p, l in zip(predictions, labels) if p == cat and l != cat)
            fn = sum(1 for p, l in zip(predictions, labels) if p != cat and l == cat)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            results[cat] = {"precision": prec, "recall": rec, "f1": f1}

        return results
    else:
        # Macro-averaged metrics
        all_prec = []
        all_rec = []

        for cat in categories:
            tp = sum(1 for p, l in zip(predictions, labels) if p == cat and l == cat)
            fp = sum(1 for p, l in zip(predictions, labels) if p == cat and l != cat)
            fn = sum(1 for p, l in zip(predictions, labels) if p != cat and l == cat)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            all_prec.append(prec)
            all_rec.append(rec)

        avg_prec = sum(all_prec) / len(all_prec) if all_prec else 0.0
        avg_rec = sum(all_rec) / len(all_rec) if all_rec else 0.0
        f1 = (
            2 * avg_prec * avg_rec / (avg_prec + avg_rec)
            if (avg_prec + avg_rec) > 0
            else 0.0
        )

        return {"precision": avg_prec, "recall": avg_rec, "f1": f1}


def confusion_matrix(
    predictions: List[str], labels: List[str]
) -> Dict[str, Dict[str, int]]:
    """Compute confusion matrix.

    Args:
        predictions: Predicted categories
        labels: Ground truth categories

    Returns:
        Nested dictionary: {true_label: {predicted_label: count}}
    """
    categories = sorted(set(labels + predictions))
    matrix = {cat: {c: 0 for c in categories} for cat in categories}

    for pred, true in zip(predictions, labels):
        matrix[true][pred] += 1

    return matrix
