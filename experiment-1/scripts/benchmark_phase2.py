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


def cmd_retrieval(args):
    """Run retrieval benchmark."""
    from hierarchical_pipeline.embedding import DummyEmbedding
    from hierarchical_pipeline.evaluation.retrieval import (
        PartRetrievalEngine,
        precision_at_k,
        recall_at_k,
        mean_average_precision,
    )
    from hierarchical_pipeline.evaluation.datasets import load_dataset
    from hierarchical_pipeline.adapters.segmentation import SegmentationDiscoveryAdapter
    from segmentation_pipeline.models import get_model, ModelConfig
    from PIL import Image
    import numpy as np

    print(f"Running retrieval benchmark: {args.dataset} dataset")

    # Load dataset
    if args.dataset == "custom":
        dataset = load_dataset(
            "custom", image_dir=args.images, num_images=args.num_images
        )
    else:
        dataset = load_dataset(args.dataset, num_images=args.num_images)

    print(f"Loaded {len(dataset)} images")

    # Setup models
    model_cfg = ModelConfig(device="cpu", model_type=None)
    seg_model = get_model(args.model, model_cfg)

    embedding_method = DummyEmbedding({"embedding_dim": 768})

    # Process all images and collect parts
    all_parts = []

    with seg_model as model:
        adapter = SegmentationDiscoveryAdapter(model)

        for img_path, _ in dataset:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)

            parts = adapter.discover_parts(img)

            # Generate embeddings
            for p in parts:
                p.embedding = embedding_method.embed_part(img_np, p.mask)

            all_parts.extend(parts)

    print(f"Collected {len(all_parts)} parts total")

    # Build index
    start_time = time.time()
    engine = PartRetrievalEngine(embedding_dim=768)
    engine.add_parts(all_parts)
    index_time = time.time() - start_time

    # Run queries
    num_queries = min(10, len(all_parts))
    query_parts = all_parts[:num_queries]

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
            "embedding": "dummy",
            "top_k": args.top_k,
        },
        "metrics": {
            "avg_precision_at_k": avg_precision,
            "avg_recall_at_k": avg_recall,
            "num_parts": len(all_parts),
            "num_queries": num_queries,
        },
        "timing": {
            "index_build_sec": index_time,
            "total_query_sec": query_time,
            "avg_query_ms": (query_time / num_queries) * 1000 if num_queries > 0 else 0,
        },
    }

    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    print("\nResults:")
    print(json.dumps(results, indent=2))

    return results


def cmd_classify(args):
    """Run classification benchmark."""
    from hierarchical_pipeline.evaluation.classification import (
        ZeroShotClassifier,
        accuracy,
    )

    print("Classification benchmark not fully implemented yet")
    print("Placeholder: would classify parts into categories and measure accuracy")

    results = {
        "config": {"dataset": args.dataset},
        "metrics": {"accuracy": 0.0},
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2B Benchmark")
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
    p_retrieval.add_argument("--output", help="Output JSON file")
    p_retrieval.set_defaults(func=cmd_retrieval)

    # Classification benchmark
    p_classify = subparsers.add_parser("classify", help="Run classification benchmark")
    p_classify.add_argument("--dataset", default="custom", help="Dataset type")
    p_classify.add_argument("--images", default="../images", help="Image directory")
    p_classify.add_argument(
        "--num-images", type=int, default=10, help="Number of images"
    )
    p_classify.add_argument("--output", help="Output JSON file")
    p_classify.set_defaults(func=cmd_classify)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
