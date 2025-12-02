"""Evaluation framework for hierarchical image representation.

Provides tools for evaluating the quality of learned hierarchies and embeddings:
- Query retrieval (finding similar parts)
- Zero-shot classification
- Dataset utilities
"""

from .retrieval import PartRetrievalEngine
from .classification import ZeroShotClassifier

__all__ = [
    "PartRetrievalEngine",
    "ZeroShotClassifier",
]
