"""Orchestrates full experiment pipeline with tracking.

This module provides the main experiment runner that coordinates segmentation,
hierarchy building, embedding generation, and optional evaluation, all with
comprehensive tracking and metadata capture.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import numpy as np

from .experiment_tracker import ExperimentTracker, ExperimentRun
from ..config import PipelineConfig
from ..adapters.segmentation import SegmentationDiscoveryAdapter
from ..core.builder import BottomUpHierarchyBuilder
from ..core.data import ParseGraph
from ..core.analysis import HierarchyMetrics
from ..visualization import (
    plot_interactive_graph,
    overlay_masks,
    overlay_masks_with_ids,
)
from segmentation_pipeline.models import get_model, ModelConfig

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates full pipeline execution with experiment tracking."""

    def __init__(
        self, config: Dict[str, Any], tracker: Optional[ExperimentTracker] = None
    ):
        """Initialize experiment runner.

        Args:
            config: Full experiment configuration
            tracker: ExperimentTracker instance (creates default if None)
        """
        self.config = config
        self.tracker = tracker or ExperimentTracker(
            experiments_dir=Path.cwd() / "experiments"
        )
        self.current_run: Optional[ExperimentRun] = None

    def run_full_pipeline(self) -> ExperimentRun:
        """Run complete pipeline with tracking.

        Returns:
            ExperimentRun with all results
        """
        start_time = time.time()

        # Extract config sections
        exp_config = self.config.get("experiment", {})
        data_config = self.config.get("data", {})
        seg_config = self.config.get("segmentation", {})
        emb_config = self.config.get("embedding", {})
        hier_config = self.config.get("hierarchy", {})
        eval_config = self.config.get("evaluation", {})
        output_config = self.config.get("output", {})
        gpu_config = self.config.get("gpu", {})

        # Determine device
        device = gpu_config.get("device") or self._auto_detect_device()

        # Start tracking
        experiment_name = exp_config.get("name", "experiment")
        self.current_run = self.tracker.start_run(
            name=experiment_name,
            config=self.config,
            device=device,
            command=None,  # Could capture from sys.argv
        )

        try:
            # Get output directory from run
            output_dir = Path(self.current_run.output_dir)
            graphs_dir = output_dir / "graphs"
            viz_dir = output_dir / "visualizations"
            graphs_dir.mkdir(exist_ok=True)
            viz_dir.mkdir(exist_ok=True)

            # Load images
            images_path = Path(data_config["images_path"])
            image_files = self._load_image_list(
                images_path, data_config.get("max_images")
            )

            logger.info(f"Processing {len(image_files)} images")

            # Setup segmentation model
            model_cfg = ModelConfig(
                device=device,
                checkpoint=seg_config.get("checkpoint"),
                model_type=seg_config.get("model_type"),
                confidence_threshold=seg_config.get("min_confidence", 0.5),
                min_area=seg_config.get("min_area", 100),
                extra_params={},
            )

            seg_model = get_model(seg_config["model"], model_cfg)

            # Setup embedding method
            embedding_method = None
            if emb_config.get("method"):
                embedding_method = self._create_embedding_method(
                    emb_config["method"], device, emb_config
                )

            # Setup hierarchy builder
            builder = BottomUpHierarchyBuilder(config={"params": hier_config})

            # Process images
            all_parts = []
            all_graphs = []

            with seg_model as model:
                adapter = SegmentationDiscoveryAdapter(
                    model,
                    config={
                        "min_confidence": seg_config.get("min_confidence", 0.5),
                        "min_area": seg_config.get("min_area", 100),
                    },
                )

                for i, img_path in enumerate(image_files):
                    logger.info(
                        f"[{i+1}/{len(image_files)}] Processing {img_path.name}"
                    )

                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_np = np.array(img)

                        # Discover parts
                        parts = adapter.discover_parts(img)

                        # Add metadata
                        for p in parts:
                            p.metadata.setdefault("image_path", str(img_path))
                            p.metadata.setdefault("image_size", (img.width, img.height))

                        # Generate embeddings
                        if embedding_method:
                            for p in parts:
                                p.embedding = embedding_method.embed_part(
                                    img_np, p.mask
                                )

                        # Build hierarchy
                        graph = builder.build_hierarchy(parts)

                        # Validate
                        errors = graph.validate()
                        if errors:
                            logger.warning(f"Graph validation warnings: {errors}")

                        # Save graph
                        if output_config.get("save_parse_graphs", True):
                            graph_path = graphs_dir / f"{img_path.stem}.graph.json"
                            with open(graph_path, "w") as f:
                                f.write(graph.to_json())

                        # Save visualizations
                        if output_config.get("save_visualizations", True):
                            self._save_visualizations(
                                graph, parts, img_np, img_path.stem, viz_dir
                            )

                        all_parts.extend(parts)
                        all_graphs.append(graph)

                    except Exception as e:
                        logger.error(f"Failed to process {img_path}: {e}")
                        continue

            # Save embeddings
            if embedding_method and output_config.get("save_embeddings", True):
                self._save_embeddings(all_parts, output_dir / "embeddings.npz")

            # Compute metrics
            metrics = self._compute_metrics(all_graphs)

            # Run evaluation if enabled
            if eval_config.get("enabled", False):
                eval_metrics = self._run_evaluation(
                    all_parts, eval_config, output_dir / "evaluation"
                )
                metrics.update(eval_metrics)

            # Complete run
            duration = time.time() - start_time
            self.tracker.complete_run(
                self.current_run,
                metrics=metrics,
                duration=duration,
                num_images=len(image_files),
                num_parts=len(all_parts),
            )

            logger.info(f"Experiment completed in {duration:.2f}s")
            logger.info(f"Results saved to: {output_dir}")

            return self.current_run

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            if self.current_run:
                self.tracker.fail_run(self.current_run, str(e))
            raise

    def _load_image_list(
        self, images_path: Path, max_images: Optional[int]
    ) -> List[Path]:
        """Load list of images to process."""
        if images_path.is_file():
            # Single image
            images = [images_path]
        elif images_path.is_dir():
            # Directory of images
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            images = [
                img
                for img in sorted(images_path.iterdir())
                if img.suffix.lower() in extensions
            ]
        else:
            raise ValueError(f"Images path not found: {images_path}")

        # Limit if quick mode or max_images specified
        if max_images:
            images = images[:max_images]

        return images

    def _create_embedding_method(
        self, method_name: str, device: str, emb_config: Dict[str, Any]
    ):
        """Create embedding method instance."""
        from ..embedding import (
            DummyEmbedding,
            DINOEmbedding,
            DINOv2Embedding,
            CLIPEmbedding,
            MAEEmbedding,
        )

        config = {
            "device": device,
            "cache_dir": emb_config.get("cache_dir", "cache/embeddings"),
            "use_cache": emb_config.get("use_cache", True),
        }

        if method_name.lower() == "dummy":
            return DummyEmbedding({"embedding_dim": 768})
        elif method_name.lower() == "dino":
            return DINOEmbedding(config)
        elif method_name.lower() == "dinov2":
            # Support model size configuration
            config["model_size"] = emb_config.get("model_size", "base")
            return DINOv2Embedding(config)
        elif method_name.lower() == "clip":
            return CLIPEmbedding(config)
        elif method_name.lower() == "mae":
            return MAEEmbedding(config)
        else:
            raise ValueError(f"Unknown embedding method: {method_name}")

    def _save_visualizations(
        self,
        graph: ParseGraph,
        parts: List,
        img_np: np.ndarray,
        stem: str,
        viz_dir: Path,
    ) -> None:
        """Save various visualizations."""
        # Interactive graph
        try:
            html_path = viz_dir / f"{stem}.graph.html"
            plot_interactive_graph(graph, output_path=str(html_path))
        except Exception as e:
            logger.warning(f"Failed to create interactive graph: {e}")

        # Mask overlays
        if parts:
            try:
                from PIL import Image as PILImage

                # Baseline segmentation
                masks = [p.mask for p in parts]
                baseline = overlay_masks(img_np, masks)
                baseline_pil = PILImage.fromarray(baseline)
                baseline_pil.save(viz_dir / f"{stem}.baseline.png")

                # Hierarchy overlay with IDs
                hierarchy_viz = overlay_masks_with_ids(img_np, parts)
                hierarchy_viz.save(viz_dir / f"{stem}.hierarchy_overlay.png")
            except Exception as e:
                logger.warning(f"Failed to create mask overlays: {e}")

    def _save_embeddings(self, parts: List, output_path: Path) -> None:
        """Save embeddings to compressed numpy format."""
        embeddings = [
            p.embedding
            for p in parts
            if hasattr(p, "embedding") and p.embedding is not None
        ]

        if embeddings:
            np.savez_compressed(
                output_path,
                embeddings=np.stack(embeddings),
                part_ids=[
                    p.id
                    for p in parts
                    if hasattr(p, "embedding") and p.embedding is not None
                ],
            )
            logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")

    def _compute_metrics(self, graphs: List[ParseGraph]) -> Dict[str, Any]:
        """Compute aggregate metrics across all graphs."""
        if not graphs:
            return {}

        # Collect metrics from each graph
        all_metrics = [HierarchyMetrics.get_all_metrics(g) for g in graphs]

        # Aggregate
        aggregated = {}
        if all_metrics:
            keys = all_metrics[0].keys()
            for key in keys:
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    aggregated[f"avg_{key}"] = sum(values) / len(values)
                    aggregated[f"min_{key}"] = min(values)
                    aggregated[f"max_{key}"] = max(values)

        aggregated["num_graphs"] = len(graphs)

        return aggregated

    def _run_evaluation(
        self, parts: List, eval_config: Dict[str, Any], eval_dir: Path
    ) -> Dict[str, Any]:
        """Run evaluation tasks."""
        eval_dir.mkdir(exist_ok=True)
        metrics = {}

        tasks = eval_config.get("tasks", [])

        if "retrieval" in tasks:
            retrieval_metrics = self._run_retrieval_eval(
                parts, eval_config.get("top_k", 5), eval_dir
            )
            metrics.update(retrieval_metrics)

        # Can add classification evaluation here

        return metrics

    def _run_retrieval_eval(
        self, parts: List, top_k: int, eval_dir: Path
    ) -> Dict[str, Any]:
        """Run retrieval evaluation."""
        from ..evaluation.retrieval import (
            PartRetrievalEngine,
            precision_at_k,
            recall_at_k,
        )

        # Filter parts with embeddings
        valid_parts = [
            p for p in parts if hasattr(p, "embedding") and p.embedding is not None
        ]

        if not valid_parts:
            logger.warning("No parts with embeddings for retrieval evaluation")
            return {}

        # Build index
        embedding_dim = valid_parts[0].embedding.shape[0]
        engine = PartRetrievalEngine(embedding_dim=embedding_dim)
        engine.add_parts(valid_parts)

        # Run queries
        num_queries = min(10, len(valid_parts))
        query_parts = valid_parts[:num_queries]

        precisions = []
        recalls = []

        for qp in query_parts:
            results = engine.query(qp, top_k=top_k)
            prec = precision_at_k(results, [qp], k=top_k)
            rec = recall_at_k(results, [qp], k=top_k)
            precisions.append(prec)
            recalls.append(rec)

        metrics = {
            f"retrieval_precision@{top_k}": (
                sum(precisions) / len(precisions) if precisions else 0.0
            ),
            f"retrieval_recall@{top_k}": (
                sum(recalls) / len(recalls) if recalls else 0.0
            ),
            "num_retrieval_queries": num_queries,
        }

        # Save detailed results
        import json

        results_path = eval_dir / "retrieval_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def _auto_detect_device(self) -> str:
        """Auto-detect available device."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
