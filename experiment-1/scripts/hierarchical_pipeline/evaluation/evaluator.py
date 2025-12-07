"""Consolidated evaluation framework for all tasks.

Provides unified interface for running retrieval, classification, and other
evaluation tasks with consistent metric reporting.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluationTask:
    """Base class for evaluation tasks."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluation task.

        Args:
            config: Task-specific configuration
        """
        self.config = config

    def run(self, parts: List[Any]) -> Dict[str, Any]:
        """Run evaluation task.

        Args:
            parts: List of Part objects with embeddings

        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError


class RetrievalEvaluator(EvaluationTask):
    """Part retrieval evaluation task."""

    def run(self, parts: List[Any]) -> Dict[str, Any]:
        """Run retrieval evaluation.

        Args:
            parts: List of Part objects with embeddings

        Returns:
            Retrieval metrics (precision@k, recall@k, etc.)
        """
        from .retrieval import PartRetrievalEngine, precision_at_k, recall_at_k

        # Filter valid parts
        valid_parts = [
            p for p in parts if hasattr(p, "embedding") and p.embedding is not None
        ]

        if not valid_parts:
            logger.warning("No parts with embeddings for retrieval")
            return {}

        top_k = self.config.get("top_k", 5)

        # Build index
        embedding_dim = valid_parts[0].embedding.shape[0]
        engine = PartRetrievalEngine(embedding_dim=embedding_dim)

        start_index = time.time()
        engine.add_parts(valid_parts)
        index_time = time.time() - start_index

        # Run queries
        num_queries = min(self.config.get("num_queries", 10), len(valid_parts))
        query_parts = valid_parts[:num_queries]

        precisions = []
        recalls = []

        start_query = time.time()
        for qp in query_parts:
            results = engine.query(qp, top_k=top_k)
            prec = precision_at_k(results, [qp], k=top_k)
            rec = recall_at_k(results, [qp], k=top_k)
            precisions.append(prec)
            recalls.append(rec)
        query_time = time.time() - start_query

        metrics = {
            f"retrieval_precision@{top_k}": (
                sum(precisions) / len(precisions) if precisions else 0.0
            ),
            f"retrieval_recall@{top_k}": (
                sum(recalls) / len(recalls) if recalls else 0.0
            ),
            "retrieval_index_time": index_time,
            "retrieval_query_time": query_time,
            "retrieval_avg_query_ms": (
                (query_time / num_queries * 1000) if num_queries > 0 else 0.0
            ),
            "retrieval_num_queries": num_queries,
            "retrieval_num_parts": len(valid_parts),
        }

        return metrics


class ClassificationEvaluator(EvaluationTask):
    """Zero-shot classification evaluation task."""

    def run(self, parts: List[Any]) -> Dict[str, Any]:
        """Run classification evaluation.

        Args:
            parts: List of Part objects with embeddings

        Returns:
            Classification metrics (accuracy, per-class metrics)
        """
        from .classification import ZeroShotClassifier

        # Filter valid parts
        valid_parts = [
            p for p in parts if hasattr(p, "embedding") and p.embedding is not None
        ]

        if not valid_parts:
            logger.warning("No parts with embeddings for classification")
            return {}

        # Get class labels
        class_labels = self.config.get(
            "class_labels",
            ["person", "vehicle", "animal", "object", "building", "furniture", "food"],
        )

        # Initialize classifier
        embedding_dim = valid_parts[0].embedding.shape[0]
        classifier = ZeroShotClassifier(embedding_dim=embedding_dim)

        # Classify
        correct = 0
        total = 0
        per_class = {label: {"correct": 0, "total": 0} for label in class_labels}

        start_time = time.time()
        for part in valid_parts:
            predicted = classifier.classify(part, class_labels)
            ground_truth = part.metadata.get("ground_truth", "unknown")

            total += 1
            is_correct = (
                predicted.lower() == ground_truth.lower()
                if ground_truth != "unknown"
                else False
            )

            if is_correct:
                correct += 1

            if ground_truth in per_class:
                per_class[ground_truth]["total"] += 1
                if is_correct:
                    per_class[ground_truth]["correct"] += 1

        classify_time = time.time() - start_time

        metrics = {
            "classification_accuracy": correct / total if total > 0 else 0.0,
            "classification_correct": correct,
            "classification_total": total,
            "classification_time": classify_time,
            "classification_avg_ms": (
                (classify_time / total * 1000) if total > 0 else 0.0
            ),
        }

        # Add per-class accuracies
        for label, stats in per_class.items():
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                metrics[f"classification_{label}_accuracy"] = acc

        return metrics


class MultiTaskEvaluator:
    """Run multiple evaluation tasks and aggregate results."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-task evaluator.

        Args:
            config: Configuration with task specifications
        """
        self.config = config
        self.tasks = self._create_tasks()

    def _create_tasks(self) -> List[EvaluationTask]:
        """Create evaluation tasks from config."""
        tasks = []
        task_names = self.config.get("tasks", ["retrieval"])

        for task_name in task_names:
            if task_name == "retrieval":
                tasks.append(RetrievalEvaluator(self.config))
            elif task_name == "classification":
                tasks.append(ClassificationEvaluator(self.config))
            else:
                logger.warning(f"Unknown task: {task_name}")

        return tasks

    def run_all(self, parts: List[Any]) -> Dict[str, Any]:
        """Run all evaluation tasks.

        Args:
            parts: List of Part objects

        Returns:
            Aggregated metrics from all tasks
        """
        all_metrics = {}

        for task in self.tasks:
            try:
                logger.info(f"Running {task.__class__.__name__}...")
                metrics = task.run(parts)
                all_metrics.update(metrics)
            except Exception as e:
                logger.error(f"Task {task.__class__.__name__} failed: {e}")
                continue

        return all_metrics
