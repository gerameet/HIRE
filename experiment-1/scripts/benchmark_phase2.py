#!/usr/bin/env python3
"""Phase 2B Benchmark Script

Run benchmarks for retrieval and classification on various configurations.

Usage:
    python benchmark_phase2.py retrieval --dataset custom --images ../images --output benchmarks/retrieval.json
    python benchmark_phase2.py classify --dataset custom --images ../images --output benchmarks/classify.json
"""

import argparse
import json
import time
from pathlib import Path
import sys

# Add to path
sys.path.insert(0, str(Path(__file__).parent))


def create_embedding_method(method_name, device="cpu"):
    """Factory function to create embedding methods by name."""
    from hierarchical_pipeline.embedding import (
        DummyEmbedding,
        DINOEmbedding,
        CLIPEmbedding,
        MAEEmbedding,
    )

    config = {
        "device": device,
        "cache_dir": "cache/embeddings",
        "use_cache": True,
    }

    if method_name.lower() == "dummy":
        return DummyEmbedding({"embedding_dim": 768}), 768
    elif method_name.lower() in ["dino", "dinov2"]:
        return DINOEmbedding(config), 768
    elif method_name.lower() == "clip":
        return CLIPEmbedding(config), 512
    elif method_name.lower() == "mae":
        return MAEEmbedding(config), 768
    else:
        raise ValueError(f"Unknown embedding method: {method_name}")


def run_single_embedding_benchmark(
    seg_model, dataset, embedding_method, embedding_name, device, args
):
    """Run benchmark for a single embedding method."""
    from hierarchical_pipeline.evaluation.retrieval import (
        PartRetrievalEngine,
        precision_at_k,
        PartRetrievalEngine,
        precision_at_k,
        recall_at_k,
    )
    from hierarchical_pipeline.adapters.segmentation import SegmentationDiscoveryAdapter
    from PIL import Image
    import numpy as np

    print(f"\n  → Running benchmark for embedding: {embedding_name}")

    # Process all images and collect parts
    all_parts = []

    with seg_model as model:
        adapter = SegmentationDiscoveryAdapter(model)

        for img_path, _ in dataset:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)

            parts = adapter.discover_parts(img)

            # Generate embeddings

            # Generate embeddings
            for p in parts:
                try:
                    p.embedding = embedding_method.embed_part(img_np, p.mask)
                except Exception as e:
                    print(f"    Warning: Failed to embed part: {e}. Skipping part.")
                    continue

            all_parts.extend(parts)

    print(f"    Collected {len(all_parts)} parts")

    if len(all_parts) == 0:
        print(f"    Error: No parts collected for embedding {embedding_name}")
        return None

    # Get embedding dimension
    embedding_dim = all_parts[0].embedding.shape[0] if all_parts else 768

    # Build index
    print(f"    Building retrieval index...")
    start_time = time.time()
    engine = PartRetrievalEngine(embedding_dim=embedding_dim)
    engine.add_parts(all_parts)
    index_time = time.time() - start_time

    # Run queries
    num_queries = min(10, len(all_parts))
    query_parts = all_parts[:num_queries]

    print(f"    Running {num_queries} queries...")
    start_time = time.time()
    all_results = []
    for qp in query_parts:
        results = engine.query(qp, top_k=args.top_k)
        all_results.append(results)
    query_time = time.time() - start_time

    # Compute metrics (using self as ground truth for demo)
    precisions = [
        precision_at_k(res, [query_parts[i]], k=args.top_k)
        for i, res in enumerate(all_results)
    ]
    recalls = [
        recall_at_k(res, [query_parts[i]], k=args.top_k)
        for i, res in enumerate(all_results)
    ]

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

    # Results
    results = {
        "config": {
            "dataset": args.dataset,
            "num_images": len(dataset),
            "model": args.model,
            "embedding": embedding_name,
            "top_k": args.top_k,
        },
        "metrics": {
            "avg_precision_at_k": round(avg_precision, 4),
            "avg_recall_at_k": round(avg_recall, 4),
            "num_parts": len(all_parts),
            "num_queries": num_queries,
        },
        "timing": {
            "index_build_sec": round(index_time, 4),
            "total_query_sec": round(query_time, 4),
            "avg_query_ms": round(
                (query_time / num_queries) * 1000 if num_queries > 0 else 0, 2
            ),
        },
    }

    print(
        f"    ✓ Precision@{args.top_k}: {avg_precision:.4f}, Recall@{args.top_k}: {avg_recall:.4f}"
    )
    print(
        f"    ✓ Index build: {index_time:.4f}s, Avg query: {results['timing']['avg_query_ms']:.2f}ms"
    )

    return results


def cmd_retrieval(args):
    """Run retrieval benchmark."""
    from hierarchical_pipeline.evaluation.datasets import load_dataset
    from segmentation_pipeline.models import get_model, ModelConfig

    print(f"Running retrieval benchmark: {args.dataset} dataset")

    # Load dataset once
    if args.dataset == "custom":
        dataset = load_dataset(
            "custom", image_dir=args.images, num_images=args.num_images
        )
    else:
        dataset = load_dataset(args.dataset, num_images=args.num_images)

    print(f"Loaded {len(dataset)} images")

    # Setup segmentation model once
    model_cfg = ModelConfig(device=args.device or "cpu", model_type=None)
    seg_model = get_model(args.model, model_cfg)

    # Determine which embeddings to run
    if args.all_embeddings:
        embedding_methods = ["dummy", "clip", "dinov2", "mae"]
    else:
        embedding_methods = [args.embedding_method]

    print(f"\nBenchmarking {len(embedding_methods)} embedding method(s)...")

    all_results = {}
    device = args.device or "cpu"

    # Run benchmark for each embedding method
    for embedding_name in embedding_methods:
        try:
            print(
                f"\n[{embedding_methods.index(embedding_name) + 1}/{len(embedding_methods)}] Embedding: {embedding_name}"
            )
            embedding_method, embedding_dim = create_embedding_method(
                embedding_name, device=device
            )

            result = run_single_embedding_benchmark(
                seg_model, dataset, embedding_method, embedding_name, device, args
            )

            if result is not None:
                all_results[embedding_name] = result
                print(f"  ✓ {embedding_name} benchmark complete")
            else:
                print(f"  ✗ {embedding_name} benchmark failed")

        except Exception as e:
            print(f"  ✗ Error benchmarking {embedding_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save results persistently
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save combined results
        combined_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_config": {
                "dataset": args.dataset,
                "segmentation_model": args.model,
                "num_embeddings": len(all_results),
            },
            "results": all_results,
        }

        with open(output_path, "w") as f:
            json.dump(combined_results, f, indent=2)

        print(f"\n✓ Combined results saved to {output_path}")

        # Also save individual JSON files for each embedding
        output_dir = output_path.parent
        for embedding_name, result in all_results.items():
            individual_file = output_dir / f"retrieval_{embedding_name}.json"
            with open(individual_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"✓ Individual result saved to {individual_file}")

        # Generate a summary CSV for easy comparison
        csv_path = output_dir / "retrieval_summary.csv"
        generate_summary_csv(all_results, csv_path)
        print(f"✓ Summary CSV saved to {csv_path}")

    # Print summary table
    print_summary_table(all_results)

    return all_results


def generate_summary_csv(results, csv_path):
    """Generate a CSV summary for easy comparison across embeddings."""
    import csv

    if not results:
        return

    # Extract all unique metric keys
    metric_keys = set()
    for result in results.values():
        metric_keys.update(result["metrics"].keys())
        metric_keys.update(result["timing"].keys())

    metric_keys = sorted(metric_keys)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["Embedding"] + metric_keys)

        # Rows
        for embedding_name in sorted(results.keys()):
            result = results[embedding_name]
            row = [embedding_name]

            for key in metric_keys:
                if key in result["metrics"]:
                    row.append(result["metrics"][key])
                elif key in result["timing"]:
                    row.append(result["timing"][key])
                else:
                    row.append("N/A")

            writer.writerow(row)


def print_summary_table(results):
    """Print a formatted summary table of results."""
    if not results:
        print("\nNo results to display")
        return

    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY ACROSS ALL EMBEDDINGS")
    print("=" * 120)

    # Header
    header = (
        f"{'Embedding':<15} "
        f"{'Precision@K':<15} "
        f"{'Recall@K':<15} "
        f"{'Num Parts':<12} "
        f"{'Index Time (s)':<15} "
        f"{'Query Time (s)':<15} "
        f"{'Avg Query (ms)':<15}"
    )
    print(header)
    print("-" * 120)

    # Rows
    for embedding_name in sorted(results.keys()):
        result = results[embedding_name]
        metrics = result["metrics"]
        timing = result["timing"]

        row = (
            f"{embedding_name:<15} "
            f"{metrics['avg_precision_at_k']:<15.4f} "
            f"{metrics['avg_recall_at_k']:<15.4f} "
            f"{metrics['num_parts']:<12} "
            f"{timing['index_build_sec']:<15.4f} "
            f"{timing['total_query_sec']:<15.4f} "
            f"{timing['avg_query_ms']:<15.2f}"
        )
        print(row)

    print("=" * 120)


def cmd_classify(args):
    """Run classification benchmark with zero-shot classifier."""
    from hierarchical_pipeline.evaluation.datasets import load_dataset
    from hierarchical_pipeline.evaluation.classification import ZeroShotClassifier
    from hierarchical_pipeline.adapters.segmentation import SegmentationDiscoveryAdapter
    from segmentation_pipeline.models import get_model, ModelConfig
    from PIL import Image
    import numpy as np

    print(f"Running classification benchmark: {args.dataset} dataset")

    # Load dataset
    if args.dataset == "custom":
        dataset = load_dataset(
            "custom", image_dir=args.images, num_images=args.num_images
        )
    else:
        dataset = load_dataset(args.dataset, num_images=args.num_images)

    print(f"Loaded {len(dataset)} images")

    # Setup segmentation model
    model_cfg = ModelConfig(device=args.device or "cpu", model_type=None)
    seg_model = get_model(args.model, model_cfg)

    # Determine which embeddings to run
    if args.all_embeddings:
        embedding_methods = ["dummy", "clip", "dinov2", "mae"]
    else:
        embedding_methods = [args.embedding_method]

    print(
        f"\nBenchmarking classification with {len(embedding_methods)} embedding method(s)..."
    )

    all_results = {}
    device = args.device or "cpu"

    # Get class labels
    if not args.class_labels:
        # Default categories for zero-shot classification
        class_labels = [
            "person",
            "animal",
            "object",
            "vehicle",
            "building",
            "furniture",
            "food",
        ]
    else:
        class_labels = args.class_labels

    print(f"Classification categories: {', '.join(class_labels)}")

    # Run classification for each embedding method
    for embedding_name in embedding_methods:
        try:
            print(
                f"\n[{embedding_methods.index(embedding_name) + 1}/{len(embedding_methods)}] Embedding: {embedding_name}"
            )
            embedding_method, embedding_dim = create_embedding_method(
                embedding_name, device=device
            )

            # Collect parts with embeddings
            print(f"  Discovering and embedding parts...")
            all_parts = []

            with seg_model as model:
                adapter = SegmentationDiscoveryAdapter(model)

                for img_path, img_label in dataset:
                    img = Image.open(img_path).convert("RGB")
                    img_np = np.array(img)

                    parts = adapter.discover_parts(img)

                    # Generate embeddings
                    for p in parts:
                        try:
                            p.embedding = embedding_method.embed_part(img_np, p.mask)
                            p.metadata["ground_truth"] = img_label
                        except Exception as e:
                            print(f"    Warning: Failed to embed part: {e}")
                            continue

                    all_parts.extend(parts)

            print(f"  Collected {len(all_parts)} parts")

            if len(all_parts) == 0:
                print(f"  Error: No parts collected for embedding {embedding_name}")
                continue

            # Initialize classifier
            print(f"  Initializing zero-shot classifier...")
            classifier = ZeroShotClassifier(
                embedding_dim=all_parts[0].embedding.shape[0]
            )

            # Classify parts
            print(f"  Running classification on {len(all_parts)} parts...")
            start_time = time.time()
            correct_predictions = 0
            total_predictions = 0
            predictions_by_label = {
                label: {"correct": 0, "total": 0} for label in class_labels
            }

            for part in all_parts:
                # Classify the part
                predicted_label = classifier.classify(part, class_labels)
                total_predictions += 1

                # Get ground truth
                ground_truth = part.metadata.get("ground_truth", "unknown")

                # Check if correct (simplified: any match counts)
                is_correct = (
                    predicted_label.lower() == ground_truth.lower()
                    if ground_truth != "unknown"
                    else False
                )

                if is_correct:
                    correct_predictions += 1

                # Track per-label statistics
                if ground_truth in predictions_by_label:
                    predictions_by_label[ground_truth]["total"] += 1
                    if is_correct:
                        predictions_by_label[ground_truth]["correct"] += 1

            classify_time = time.time() - start_time

            # Compute metrics
            overall_accuracy = (
                correct_predictions / total_predictions
                if total_predictions > 0
                else 0.0
            )
            per_label_accuracy = {
                label: data["correct"] / data["total"] if data["total"] > 0 else 0.0
                for label, data in predictions_by_label.items()
            }

            # Results
            results = {
                "config": {
                    "dataset": args.dataset,
                    "num_images": len(dataset),
                    "segmentation_model": args.model,
                    "embedding": embedding_name,
                    "class_labels": class_labels,
                    "num_classes": len(class_labels),
                },
                "metrics": {
                    "overall_accuracy": round(overall_accuracy, 4),
                    "correct_predictions": correct_predictions,
                    "total_predictions": total_predictions,
                    "per_label_accuracy": {
                        k: round(v, 4) for k, v in per_label_accuracy.items()
                    },
                    "num_parts": len(all_parts),
                },
                "timing": {
                    "classification_sec": round(classify_time, 4),
                    "avg_prediction_ms": round(
                        (
                            (classify_time / total_predictions) * 1000
                            if total_predictions > 0
                            else 0
                        ),
                        2,
                    ),
                },
            }

            all_results[embedding_name] = results
            print(f"  ✓ Overall accuracy: {overall_accuracy:.4f}")
            print(
                f"  ✓ Classification time: {classify_time:.4f}s ({results['timing']['avg_prediction_ms']:.2f}ms per sample)"
            )

        except Exception as e:
            print(f"  ✗ Error classifying with {embedding_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save results persistently
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save combined results
        combined_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_config": {
                "dataset": args.dataset,
                "segmentation_model": args.model,
                "num_embeddings": len(all_results),
                "class_labels": class_labels,
            },
            "results": all_results,
        }

        with open(output_path, "w") as f:
            json.dump(combined_results, f, indent=2)

        print(f"\n✓ Combined results saved to {output_path}")

        # Save individual JSON files for each embedding
        output_dir = output_path.parent
        for embedding_name, result in all_results.items():
            individual_file = output_dir / f"classification_{embedding_name}.json"
            with open(individual_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"✓ Individual result saved to {individual_file}")

        # Generate summary CSV
        csv_path = output_dir / "classification_summary.csv"
        generate_classification_csv(all_results, csv_path)
        print(f"✓ Summary CSV saved to {csv_path}")

    # Print summary table
    print_classification_summary_table(all_results)

    return all_results


def generate_classification_csv(results, csv_path):
    """Generate CSV summary for classification results."""
    import csv

    if not results:
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "Embedding",
                "Overall Accuracy",
                "Correct Predictions",
                "Total Predictions",
                "Classification Time (s)",
                "Avg Prediction (ms)",
            ]
        )

        # Rows
        for embedding_name in sorted(results.keys()):
            result = results[embedding_name]
            metrics = result["metrics"]
            timing = result["timing"]

            writer.writerow(
                [
                    embedding_name,
                    metrics["overall_accuracy"],
                    metrics["correct_predictions"],
                    metrics["total_predictions"],
                    timing["classification_sec"],
                    timing["avg_prediction_ms"],
                ]
            )


def print_classification_summary_table(results):
    """Print formatted classification summary table."""
    if not results:
        print("\nNo results to display")
        return

    print("\n" + "=" * 100)
    print("CLASSIFICATION BENCHMARK SUMMARY")
    print("=" * 100)

    # Header
    header = (
        f"{'Embedding':<15} "
        f"{'Accuracy':<12} "
        f"{'Correct':<10} "
        f"{'Total':<10} "
        f"{'Time (s)':<12} "
        f"{'Avg Pred (ms)':<15}"
    )
    print(header)
    print("-" * 100)

    # Rows
    for embedding_name in sorted(results.keys()):
        result = results[embedding_name]
        metrics = result["metrics"]
        timing = result["timing"]

        row = (
            f"{embedding_name:<15} "
            f"{metrics['overall_accuracy']:<12.4f} "
            f"{metrics['correct_predictions']:<10} "
            f"{metrics['total_predictions']:<10} "
            f"{timing['classification_sec']:<12.4f} "
            f"{timing['avg_prediction_ms']:<15.2f}"
        )
        print(row)

    print("=" * 100)


def cmd_ablation_study(args):
    """Run comprehensive ablation study: all seg models × all embeddings."""
    print("=" * 120)
    print("COMPREHENSIVE ABLATION STUDY")
    print("Testing all combinations of segmentation and embedding models")
    print("=" * 120)

    # Get available models
    from segmentation_pipeline.models import list_available_models

    seg_models = [m for m in list_available_models() if m != "dummy"]  # Exclude dummy
    embedding_methods = ["clip", "dinov2", "mae"]  # Exclude dummy
    benchmark_type = args.benchmark_type  # retrieval or classify

    print(f"\nSegmentation models: {len(seg_models)} → {', '.join(seg_models)}")
    print(
        f"Embedding methods: {len(embedding_methods)} → {', '.join(embedding_methods)}"
    )
    print(f"Benchmark type: {benchmark_type}")
    print(f"Total combinations to run: {len(seg_models) * len(embedding_methods)}")

    all_results = {}
    combination_count = 0
    failed_combinations = []

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for seg_model in seg_models:
        seg_model_results = {}

        for embedding_method in embedding_methods:
            combination_count += 1
            combo_name = f"{seg_model}_{embedding_method}"

            print(
                f"\n[{combination_count}/{len(seg_models) * len(embedding_methods)}] {combo_name}"
            )
            print(f"  Segmentation: {seg_model}")
            print(f"  Embedding: {embedding_method}")

            try:
                # Prepare arguments for the benchmark
                if benchmark_type == "retrieval":
                    # Run retrieval benchmark
                    args_copy = argparse.Namespace(
                        dataset=args.dataset,
                        images=args.images,
                        num_images=args.num_images,
                        model=seg_model,
                        embedding_method=embedding_method,
                        all_embeddings=False,
                        top_k=args.top_k,
                        device=args.device,
                        output=None,  # Don't save individual files
                    )
                    result = run_retrieval_benchmark_single_combo(args_copy)
                else:
                    # Run classification benchmark
                    args_copy = argparse.Namespace(
                        dataset=args.dataset,
                        images=args.images,
                        num_images=args.num_images,
                        model=seg_model,
                        embedding_method=embedding_method,
                        all_embeddings=False,
                        device=args.device,
                        class_labels=args.class_labels,
                        output=None,  # Don't save individual files
                    )
                    result = run_classification_benchmark_single_combo(args_copy)

                if result is not None:
                    seg_model_results[embedding_method] = result
                    print(f"  ✓ Success")
                else:
                    print(f"  ✗ Failed (no result)")
                    failed_combinations.append(combo_name)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                failed_combinations.append(combo_name)
                continue

        if seg_model_results:
            all_results[seg_model] = seg_model_results

    # Save comprehensive ablation results
    if args.output:
        output_path = Path(args.output)

        # Combined ablation JSON
        ablation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmark_type": benchmark_type,
            "global_config": {
                "dataset": args.dataset,
                "num_images": args.num_images,
                "segmentation_models": seg_models,
                "embedding_methods": embedding_methods,
                "total_combinations": len(seg_models) * len(embedding_methods),
                "successful_runs": combination_count - len(failed_combinations),
                "failed_runs": len(failed_combinations),
            },
            "results": all_results,
        }

        with open(output_path, "w") as f:
            json.dump(ablation_results, f, indent=2)

        print(f"\n✓ Ablation study results saved to {output_path}")

        # Generate comprehensive ablation CSV
        csv_path = output_dir / f"ablation_{benchmark_type}_summary.csv"
        generate_ablation_csv(all_results, csv_path, benchmark_type)
        print(f"✓ Ablation summary CSV saved to {csv_path}")

        # Generate HTML report
        html_path = output_dir / f"ablation_{benchmark_type}_report.html"
        generate_ablation_html_report(
            all_results, html_path, benchmark_type, seg_models, embedding_methods
        )
        print(f"✓ Ablation HTML report saved to {html_path}")

    # Print summary
    print_ablation_summary(all_results, benchmark_type, failed_combinations)

    return all_results


def run_retrieval_benchmark_single_combo(args):
    """Run a single retrieval benchmark combination (internal use)."""
    from hierarchical_pipeline.evaluation.retrieval import (
        PartRetrievalEngine,
        precision_at_k,
        recall_at_k,
    )
    from hierarchical_pipeline.evaluation.datasets import load_dataset
    from hierarchical_pipeline.adapters.segmentation import SegmentationDiscoveryAdapter
    from segmentation_pipeline.models import get_model, ModelConfig
    from PIL import Image
    import numpy as np

    try:
        # Load dataset
        if args.dataset == "custom":
            dataset = load_dataset(
                "custom", image_dir=args.images, num_images=args.num_images
            )
        else:
            dataset = load_dataset(args.dataset, num_images=args.num_images)

        # Setup models
        model_cfg = ModelConfig(device=args.device or "cpu", model_type=None)
        seg_model = get_model(args.model, model_cfg)
        embedding_method, embedding_dim = create_embedding_method(
            args.embedding_method, device=args.device or "cpu"
        )

        # Process images
        all_parts = []
        with seg_model as model:
            adapter = SegmentationDiscoveryAdapter(model)

            for img_path, _ in dataset:
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)
                parts = adapter.discover_parts(img)

                for p in parts:
                    try:
                        p.embedding = embedding_method.embed_part(img_np, p.mask)
                    except:
                        continue

                all_parts.extend(parts)

        if len(all_parts) == 0:
            return None

        # Build index and run benchmark
        engine = PartRetrievalEngine(embedding_dim=embedding_dim)
        engine.add_parts(all_parts)

        num_queries = min(10, len(all_parts))
        query_parts = all_parts[:num_queries]

        all_benchmark_results = []
        for qp in query_parts:
            results = engine.query(qp, top_k=args.top_k)
            all_benchmark_results.append(results)

        precisions = [
            precision_at_k(res, [query_parts[i]], k=args.top_k)
            for i, res in enumerate(all_benchmark_results)
        ]
        recalls = [
            recall_at_k(res, [query_parts[i]], k=args.top_k)
            for i, res in enumerate(all_benchmark_results)
        ]

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

        return {
            "metrics": {
                "avg_precision_at_k": round(avg_precision, 4),
                "avg_recall_at_k": round(avg_recall, 4),
                "num_parts": len(all_parts),
            }
        }

    except Exception as e:
        print(f"    Retrieval benchmark failed: {e}")
        return None


def run_classification_benchmark_single_combo(args):
    """Run a single classification benchmark combination (internal use)."""
    from hierarchical_pipeline.evaluation.datasets import load_dataset
    from hierarchical_pipeline.evaluation.classification import ZeroShotClassifier
    from hierarchical_pipeline.adapters.segmentation import SegmentationDiscoveryAdapter
    from segmentation_pipeline.models import get_model, ModelConfig
    from PIL import Image
    import numpy as np

    try:
        # Load dataset
        if args.dataset == "custom":
            dataset = load_dataset(
                "custom", image_dir=args.images, num_images=args.num_images
            )
        else:
            dataset = load_dataset(args.dataset, num_images=args.num_images)

        # Setup models
        model_cfg = ModelConfig(device=args.device or "cpu", model_type=None)
        seg_model = get_model(args.model, model_cfg)
        embedding_method, embedding_dim = create_embedding_method(
            args.embedding_method, device=args.device or "cpu"
        )

        # Get class labels
        class_labels = (
            args.class_labels
            if args.class_labels
            else ["person", "animal", "object", "vehicle", "building"]
        )

        # Process images
        all_parts = []
        with seg_model as model:
            adapter = SegmentationDiscoveryAdapter(model)

            for img_path, img_label in dataset:
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)
                parts = adapter.discover_parts(img)

                for p in parts:
                    try:
                        p.embedding = embedding_method.embed_part(img_np, p.mask)
                        p.metadata["ground_truth"] = img_label
                    except:
                        continue

                all_parts.extend(parts)

        if len(all_parts) == 0:
            return None

        # Classify parts
        classifier = ZeroShotClassifier(embedding_dim=embedding_dim)
        correct_predictions = 0
        total_predictions = 0

        for part in all_parts:
            predicted_label = classifier.classify(part, class_labels)
            ground_truth = part.metadata.get("ground_truth", "unknown")
            is_correct = (
                predicted_label.lower() == ground_truth.lower()
                if ground_truth != "unknown"
                else False
            )

            if is_correct:
                correct_predictions += 1
            total_predictions += 1

        overall_accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        )

        return {
            "metrics": {
                "overall_accuracy": round(overall_accuracy, 4),
                "correct_predictions": correct_predictions,
                "total_predictions": total_predictions,
            }
        }

    except Exception as e:
        print(f"    Classification benchmark failed: {e}")
        return None


def generate_ablation_csv(results, csv_path, benchmark_type):
    """Generate ablation study CSV matrix."""
    import csv

    if not results:
        return

    # Determine metric to display based on benchmark type
    if benchmark_type == "retrieval":
        metric_name = "avg_precision_at_k"
    else:
        metric_name = "overall_accuracy"

    # Get all embedding methods
    embedding_methods = []
    for seg_results in results.values():
        embedding_methods.extend(seg_results.keys())
    embedding_methods = sorted(set(embedding_methods))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["Segmentation Model"] + embedding_methods)

        # Rows (one per segmentation model)
        for seg_model in sorted(results.keys()):
            seg_results = results[seg_model]
            row = [seg_model]

            for embedding_method in embedding_methods:
                if embedding_method in seg_results:
                    value = seg_results[embedding_method]["metrics"].get(
                        metric_name, "N/A"
                    )
                    row.append(value)
                else:
                    row.append("N/A")

            writer.writerow(row)


def generate_ablation_html_report(
    results, html_path, benchmark_type, seg_models, embedding_methods
):
    """Generate comprehensive HTML ablation report."""

    # Determine metric names
    if benchmark_type == "retrieval":
        metric_name = "avg_precision_at_k"
        metric_display = "Precision@K"
    else:
        metric_name = "overall_accuracy"
        metric_display = "Accuracy"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ablation Study Report - {benchmark_type.upper()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; background-color: white; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background-color: #4CAF50; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f0f0f0; }}
        .summary {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .high {{ background-color: #c8e6c9; }}
        .medium {{ background-color: #fff9c4; }}
        .low {{ background-color: #ffccbc; }}
        .metric {{ font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Ablation Study Report</h1>
    <h2>Benchmark Type: {benchmark_type.upper()}</h2>
    
    <div class="summary">
        <h3>Summary</h3>
        <p><strong>Metric:</strong> {metric_display}</p>
        <p><strong>Segmentation Models:</strong> {len(seg_models)} models tested</p>
        <p><strong>Embedding Methods:</strong> {len(embedding_methods)} methods tested</p>
        <p><strong>Total Combinations:</strong> {len(seg_models) * len(embedding_methods)}</p>
    </div>

    <h2>Results Matrix</h2>
    <table>
        <tr>
            <th>Segmentation Model</th>
            {"".join([f"<th>{method}</th>" for method in embedding_methods])}
        </tr>
"""

    # Find min/max for color coding
    all_values = []
    for seg_model in seg_models:
        if seg_model in results:
            for embedding_method in embedding_methods:
                if embedding_method in results[seg_model]:
                    value = results[seg_model][embedding_method]["metrics"].get(
                        metric_name, None
                    )
                    if value is not None:
                        all_values.append(value)

    min_val = min(all_values) if all_values else 0
    max_val = max(all_values) if all_values else 1
    mid_val = (min_val + max_val) / 2

    # Add rows
    for seg_model in sorted(results.keys()):
        html_content += f"        <tr><td><strong>{seg_model}</strong></td>"

        for embedding_method in embedding_methods:
            if embedding_method in results[seg_model]:
                value = results[seg_model][embedding_method]["metrics"].get(
                    metric_name, None
                )
                if value is not None:
                    # Color code based on value
                    if value >= mid_val:
                        css_class = "high"
                    else:
                        css_class = "low"
                    html_content += f'<td class="{css_class}">{value:.4f}</td>'
                else:
                    html_content += "<td>N/A</td>"
            else:
                html_content += "<td>N/A</td>"

        html_content += "</tr>\n"

    html_content += """
    </table>

    <h2>Best Combinations</h2>
    <ol>
"""

    # Find top 5 combinations
    combinations = []
    for seg_model in results:
        for embedding_method in results[seg_model]:
            value = results[seg_model][embedding_method]["metrics"].get(
                metric_name, None
            )
            if value is not None:
                combinations.append((f"{seg_model} + {embedding_method}", value))

    combinations.sort(key=lambda x: x[1], reverse=True)

    for i, (combo, value) in enumerate(combinations[:5], 1):
        html_content += f"        <li><strong>{combo}</strong>: {value:.4f}</li>\n"

    html_content += """
    </ol>

    <h2>Worst Combinations</h2>
    <ol>
"""

    for i, (combo, value) in enumerate(combinations[-5:], 1):
        html_content += f"        <li><strong>{combo}</strong>: {value:.4f}</li>\n"

    html_content += (
        """
    </ol>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ccc; color: #666;">
        <p>Generated on """
        + time.strftime("%Y-%m-%d %H:%M:%S")
        + """</p>
    </footer>
</body>
</html>
"""
    )

    with open(html_path, "w") as f:
        f.write(html_content)


def print_ablation_summary(results, benchmark_type, failed_combinations):
    """Print ablation study summary to console."""

    if benchmark_type == "retrieval":
        metric_name = "avg_precision_at_k"
        metric_display = "Precision@K"
    else:
        metric_name = "overall_accuracy"
        metric_display = "Accuracy"

    print("\n" + "=" * 120)
    print(f"ABLATION STUDY SUMMARY - {benchmark_type.upper()}")
    print("=" * 120)

    # Print matrix header
    seg_models = sorted(results.keys())
    embedding_methods = []
    for seg_results in results.values():
        embedding_methods.extend(seg_results.keys())
    embedding_methods = sorted(set(embedding_methods))

    # Print header
    header = f"{'Seg Model':<20}"
    for method in embedding_methods:
        header += f"{method:<15}"
    print(header)
    print("-" * 120)

    # Print rows
    for seg_model in seg_models:
        row = f"{seg_model:<20}"
        for embedding_method in embedding_methods:
            if embedding_method in results[seg_model]:
                value = results[seg_model][embedding_method]["metrics"].get(
                    metric_name, None
                )
                if value is not None:
                    row += f"{value:<15.4f}"
                else:
                    row += f"{'N/A':<15}"
            else:
                row += f"{'N/A':<15}"
        print(row)

    print("=" * 120)

    # Print statistics
    print(f"\n{metric_display} Statistics:")
    all_values = []
    for seg_results in results.values():
        for embedding_result in seg_results.values():
            value = embedding_result["metrics"].get(metric_name, None)
            if value is not None:
                all_values.append(value)

    if all_values:
        print(f"  Best:  {max(all_values):.4f}")
        print(f"  Worst: {min(all_values):.4f}")
        print(f"  Mean:  {sum(all_values) / len(all_values):.4f}")

    if failed_combinations:
        print(f"\nFailed combinations ({len(failed_combinations)}):")
        for combo in failed_combinations[:10]:
            print(f"  - {combo}")
        if len(failed_combinations) > 10:
            print(f"  ... and {len(failed_combinations) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2B Benchmark - Test retrieval and classification across embedding methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark retrieval with dummy embedding
  python benchmark_phase2.py retrieval --dataset custom --images ../images --output ../benchmarks/retrieval.json

  # Benchmark retrieval with all embeddings
  python benchmark_phase2.py retrieval --dataset custom --images ../images --all-embeddings --output ../benchmarks/retrieval_all.json

  # Benchmark classification
  python benchmark_phase2.py classify --dataset custom --images ../images --output ../benchmarks/classify.json

  # Run comprehensive ablation study (all seg models × all embeddings)
  python benchmark_phase2.py ablation-study --benchmark-type retrieval --dataset custom --images ../images --output ../benchmarks/ablation_study.json

  # Ablation with classification
  python benchmark_phase2.py ablation-study --benchmark-type classify --dataset custom --images ../images --output ../benchmarks/ablation_classify.json
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Benchmark type")

    # Retrieval benchmark
    p_retrieval = subparsers.add_parser("retrieval", help="Run retrieval benchmark")
    p_retrieval.add_argument("--dataset", default="custom", help="Dataset type")
    p_retrieval.add_argument("--images", default="../images", help="Image directory")
    p_retrieval.add_argument(
        "--num-images", type=int, default=10, help="Number of images"
    )
    p_retrieval.add_argument("--model", default="dummy", help="Segmentation model")
    p_retrieval.add_argument("--top-k", type=int, default=5, help="Top-K for metrics")
    p_retrieval.add_argument(
        "--embedding-method",
        default="dummy",
        choices=["dummy", "clip", "dinov2", "mae"],
        help="Embedding method (default: dummy)",
    )
    p_retrieval.add_argument(
        "--all-embeddings",
        action="store_true",
        help="Run benchmarks for all available embedding methods (dummy, clip, dinov2, mae)",
    )
    p_retrieval.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu, auto-detect if not specified)",
    )
    p_retrieval.add_argument("--output", help="Output JSON file")
    p_retrieval.set_defaults(func=cmd_retrieval)

    # Classification benchmark
    p_classify = subparsers.add_parser("classify", help="Run classification benchmark")
    p_classify.add_argument("--dataset", default="custom", help="Dataset type")
    p_classify.add_argument("--images", default="../images", help="Image directory")
    p_classify.add_argument(
        "--num-images", type=int, default=10, help="Number of images"
    )
    p_classify.add_argument("--model", default="dummy", help="Segmentation model")
    p_classify.add_argument(
        "--embedding-method",
        default="dummy",
        choices=["dummy", "clip", "dinov2", "mae"],
        help="Embedding method (default: dummy)",
    )
    p_classify.add_argument(
        "--all-embeddings",
        action="store_true",
        help="Run classification for all embedding methods",
    )
    p_classify.add_argument(
        "--class-labels",
        nargs="+",
        help="Class labels for classification (default: person, animal, object, vehicle, building)",
    )
    p_classify.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu, auto-detect if not specified)",
    )
    p_classify.add_argument("--output", help="Output JSON file")
    p_classify.set_defaults(func=cmd_classify)

    # Ablation study benchmark
    p_ablation = subparsers.add_parser(
        "ablation-study",
        help="Run comprehensive ablation study (all seg models × all embeddings)",
    )
    p_ablation.add_argument(
        "--benchmark-type",
        choices=["retrieval", "classify"],
        required=True,
        help="Type of benchmark to run (retrieval or classify)",
    )
    p_ablation.add_argument("--dataset", default="custom", help="Dataset type")
    p_ablation.add_argument("--images", default="../images", help="Image directory")
    p_ablation.add_argument(
        "--num-images", type=int, default=10, help="Number of images per combination"
    )
    p_ablation.add_argument(
        "--top-k", type=int, default=5, help="Top-K for retrieval metrics"
    )
    p_ablation.add_argument(
        "--class-labels",
        nargs="+",
        help="Class labels for classification ablation",
    )
    p_ablation.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu, auto-detect if not specified)",
    )
    p_ablation.add_argument(
        "--output",
        required=True,
        help="Output JSON file for ablation study results",
    )
    p_ablation.set_defaults(func=cmd_ablation_study)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
