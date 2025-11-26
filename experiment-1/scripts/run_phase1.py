#!/usr/bin/env python3
"""Run Phase 1 pipeline: segmentation -> parts -> bottom-up hierarchy -> visualization

Usage examples:
  python run_phase1.py process --images data/images --model dummy --output output/phase1
  python run_phase1.py visualize --input output/phase1 --output output/viz
  python run_phase1.py analyze --input output/phase1
  python run_phase1.py compare --input output/phase1 --models sam,mask2former

This script is intentionally minimal and uses the existing segmentation
pipeline's model registry. It writes parse graphs (JSON) and mask overlays.
"""

import argparse
import os
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Any


def add_scripts_to_path():
    # Ensure experiment-1/scripts is on sys.path so package imports work
    repo_root = Path(__file__).resolve().parents[1]
    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)


def list_images(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        return [str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in exts]
    raise ValueError(f"Images path not found: {path}")


def cmd_process(args):
    """Run the processing pipeline."""
    add_scripts_to_path()
    
    # Import local packages
    from hierarchical_pipeline.config import (
        create_default_config,
        validate_config,
    )
    from hierarchical_pipeline.utils.gpu import GPUManager
    from hierarchical_pipeline.adapters.segmentation import SegmentationDiscoveryAdapter
    from hierarchical_pipeline.core.builder import BottomUpHierarchyBuilder
    from hierarchical_pipeline.visualization import overlay_masks, plot_parse_tree, plot_interactive_graph
    from segmentation_pipeline.models import get_model, ModelConfig, list_available_models
    from hierarchical_pipeline.core.analysis import HierarchyMetrics

    if args.list_models:
        print("Available segmentation models:")
        for m in list_available_models():
            print(" - ", m)
        return

    images = list_images(args.images)
    os.makedirs(args.output, exist_ok=True)
    
    # Prepare CSV summary path
    summary_csv = Path(args.output) / "phase1_summary.csv"
    summary_rows = []

    # Load default config
    config = create_default_config()
    if args.device:
        config.gpu.device = args.device

    validate_config(config)

    gpu_mgr = GPUManager(
        device=config.gpu.device, allow_cpu_fallback=config.gpu.allow_cpu_fallback
    )
    print(f"Using device: {gpu_mgr.device}")

    # Instantiate segmentation model
    model_extra = {}
    if args.model_params:
        try:
            pth = Path(args.model_params)
            if pth.exists():
                with open(pth, "r") as _f:
                    model_extra = json.load(_f)
            else:
                model_extra = json.loads(args.model_params)
        except Exception as e:
            print(f"Failed to parse --model-params: {e}")
            return

    model_cfg = ModelConfig(
        device=str(gpu_mgr.device) if gpu_mgr.device else None,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        extra_params=model_extra
    )
    
    try:
        seg_model = get_model(args.model, model_cfg)
    except Exception as e:
        print(f"Failed to create segmentation model '{args.model}': {e}")
        return

    # Initialize model
    with seg_model as model:
        adapter = SegmentationDiscoveryAdapter(
            model,
            config={
                "min_confidence": model.config.confidence_threshold,
                "min_area": model.config.min_area,
            },
        )

        builder = BottomUpHierarchyBuilder(config=config.hierarchy.__dict__)

        from PIL import Image

        # Prepare prompts if provided
        prompts = {}
        if args.text_prompt:
            prompts["text"] = args.text_prompt

        for img_path in images:
            print(f"Processing {img_path}")
            try:
                img = Image.open(img_path).convert("RGB")
                parts = adapter.discover_parts(img, prompts=prompts)

                # attach image metadata to parts
                for p in parts:
                    p.metadata.setdefault("image_path", img_path)
                    p.metadata.setdefault("image_size", (img.width, img.height))

                graph = builder.build_hierarchy(parts)
                
                # Validate graph
                errors = graph.validate()
                if errors:
                    print(f"Warning: Graph validation errors for {img_path}: {errors}")

                # Collect summary info
                metrics = HierarchyMetrics.get_all_metrics(graph)
                summary_row = {"image": img_path, **metrics}
                summary_rows.append(summary_row)

                stem = Path(img_path).stem
                out_prefix = Path(args.output) / stem

                # Save graph
                json_path = str(out_prefix.with_suffix(".graph.json"))
                with open(json_path, "w") as f:
                    f.write(graph.to_json())
                print(f"Saved graph: {json_path}")

                if args.save_viz:
                    # Static plot
                    try:
                        overlay_img, fig = plot_parse_tree(graph, image=None, show_overlay=False)
                        if fig is not None:
                            graph_img_path = str(out_prefix.with_suffix(".graph.png"))
                            fig.savefig(graph_img_path, bbox_inches="tight")
                            import matplotlib.pyplot as plt
                            plt.close(fig)
                    except Exception as e:
                        print(f"Static plot failed: {e}")

                    # Interactive plot
                    try:
                        html_path = str(out_prefix.with_suffix(".graph.html"))
                        plot_interactive_graph(graph, output_path=html_path)
                    except Exception as e:
                        print(f"Interactive plot failed: {e}")

                    # Overlay
                    if parts:
                        import numpy as np
                        img_np = np.array(img)
                        
                        # 1. Baseline Segmentation (just masks)
                        from hierarchical_pipeline.visualization import overlay_masks_with_ids
                        masks = [p.mask for p in parts]
                        baseline = overlay_masks(img_np, masks)
                        from PIL import Image as PILImage
                        baseline_pil = PILImage.fromarray(baseline)
                        baseline_path = str(out_prefix.with_suffix(".baseline.png"))
                        baseline_pil.save(baseline_path)
                        print(f"Saved baseline: {baseline_path}")
                        
                        # 2. Hierarchical Overlay with IDs
                        # We use the same parts list but now annotated with IDs
                        # Note: In a full hierarchy, we might want to visualize different levels
                        # For now, we visualize the leaf parts (which are what we have in 'parts')
                        # and label them with their IDs.
                        hierarchy_viz = overlay_masks_with_ids(img_np, parts)
                        hierarchy_path = str(out_prefix.with_suffix(".hierarchy_overlay.png"))
                        hierarchy_viz.save(hierarchy_path)
                        print(f"Saved hierarchy overlay: {hierarchy_path}")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    # Write summary CSV
    if summary_rows:
        keys = summary_rows[0].keys()
        with open(summary_csv, "w", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved summary CSV: {summary_csv}")


def cmd_visualize(args):
    """Generate visualizations for existing graphs."""
    add_scripts_to_path()
    from hierarchical_pipeline.core.data import ParseGraph
    from hierarchical_pipeline.visualization import plot_interactive_graph
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input path not found: {input_path}")
        return
        
    files = list(input_path.glob("*.graph.json")) if input_path.is_dir() else [input_path]
    os.makedirs(args.output, exist_ok=True)
    
    for f in files:
        try:
            with open(f, "r") as fp:
                graph = ParseGraph.from_json(fp.read())
            
            out_path = Path(args.output) / f.with_suffix(".html").name
            plot_interactive_graph(graph, output_path=str(out_path))
            print(f"Generated {out_path}")
        except Exception as e:
            print(f"Failed to visualize {f}: {e}")


def cmd_analyze(args):
    """Analyze metrics for existing graphs."""
    add_scripts_to_path()
    from hierarchical_pipeline.core.data import ParseGraph
    from hierarchical_pipeline.core.analysis import HierarchyMetrics
    
    input_path = Path(args.input)
    files = list(input_path.glob("*.graph.json")) if input_path.is_dir() else [input_path]
    
    summary_rows = []
    for f in files:
        try:
            with open(f, "r") as fp:
                graph = ParseGraph.from_json(fp.read())
            metrics = HierarchyMetrics.get_all_metrics(graph)
            summary_rows.append({"file": f.name, **metrics})
        except Exception as e:
            print(f"Failed to analyze {f}: {e}")
            
    if summary_rows:
        try:
            import pandas as pd
            df = pd.DataFrame(summary_rows)
            print(df.describe())
            if args.output:
                df.to_csv(args.output, index=False)
                print(f"Saved analysis to {args.output}")
        except ImportError:
            print("Pandas not found. Printing raw summary:")
            print(json.dumps(summary_rows, indent=2))
            if args.output:
                keys = summary_rows[0].keys()
                with open(args.output, "w", newline="") as csvf:
                    writer = csv.DictWriter(csvf, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(summary_rows)
                print(f"Saved analysis to {args.output}")


def cmd_compare(args):
    """Compare results from different models."""
    add_scripts_to_path()
    from hierarchical_pipeline.core.analysis import HierarchyMetrics
    from hierarchical_pipeline.core.data import ParseGraph
    import pandas as pd
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        HAS_PLOTTING = True
    except ImportError:
        HAS_PLOTTING = False
        print("Seaborn/Matplotlib not found. Skipping plots.")

    input_dir = Path(args.inputs[0])
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return
        
    # Assume input directory structure: output/phase1/<model_name>/<image>.graph.json
    # Or maybe we just look for all graph.json files recursively and extract model from path?
    # For simplicity, let's assume the user ran 'process' multiple times with different output dirs
    # and passed a list of directories to compare.
    
    # Actually, let's change the args to accept multiple input directories
    # usage: compare --inputs output/model1 output/model2 --labels model1 model2
    
    inputs = args.inputs
    labels = args.labels
    
    if len(inputs) != len(labels):
        print("Error: Number of inputs must match number of labels")
        return
        
    data = []
    
    for inp, label in zip(inputs, labels):
        inp_path = Path(inp)
        files = list(inp_path.glob("*.graph.json"))
        for f in files:
            try:
                with open(f, "r") as fp:
                    graph = ParseGraph.from_json(fp.read())
                metrics = HierarchyMetrics.get_all_metrics(graph)
                data.append({"model": label, "image": f.stem.replace(".graph", ""), **metrics})
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    if not data:
        print("No data found to compare.")
        return
        
    df = pd.DataFrame(data)
    print("\nComparison Summary:")
    print(df.groupby("model").mean(numeric_only=True))
    
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        df.to_csv(Path(args.output) / "comparison.csv", index=False)
        
        if HAS_PLOTTING:
            # Generate comparison plots
            metrics_to_plot = ["depth", "branching_factor", "balance", "num_nodes"]
            
            for metric in metrics_to_plot:
                try:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=df, x="model", y=metric)
                    plt.title(f"Comparison of {metric}")
                    plt.savefig(Path(args.output) / f"compare_{metric}.png")
                    plt.close()
                except Exception as e:
                    print(f"Failed to plot {metric}: {e}")
            
        print(f"Saved comparison results to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="HIRE Phase 1 CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process command
    p_process = subparsers.add_parser("process", help="Run segmentation pipeline")
    p_process.add_argument("--images", required=True, help="Input images")
    p_process.add_argument("--model", default="dummy", help="Segmentation model")
    p_process.add_argument("--model-type", help="Model type (e.g. vit_h, vit_b)")
    p_process.add_argument("--checkpoint", help="Model checkpoint path or ID")
    p_process.add_argument("--text-prompt", nargs="+", help="Text prompts for models like CLIPSeg")
    p_process.add_argument("--list-models", action="store_true", help="List models")
    p_process.add_argument("--model-params", help="Model parameters (JSON)")
    p_process.add_argument("--output", default="output/phase1", help="Output directory")
    p_process.add_argument("--device", help="Device (cuda/cpu)")
    p_process.add_argument("--save-viz", action="store_true", default=True, help="Save visualizations")
    p_process.set_defaults(func=cmd_process)
    
    # Visualize command
    p_viz = subparsers.add_parser("visualize", help="Visualize existing graphs")
    p_viz.add_argument("--input", required=True, help="Input directory or file")
    p_viz.add_argument("--output", default="output/viz", help="Output directory")
    p_viz.set_defaults(func=cmd_visualize)
    
    # Analyze command
    p_analyze = subparsers.add_parser("analyze", help="Analyze metrics")
    p_analyze.add_argument("--input", required=True, help="Input directory or file")
    p_analyze.add_argument("--output", help="Output CSV file")
    p_analyze.set_defaults(func=cmd_analyze)
    
    # Compare command
    p_compare = subparsers.add_parser("compare", help="Compare models")
    p_compare.add_argument("--inputs", nargs="+", required=True, help="List of input directories")
    p_compare.add_argument("--labels", nargs="+", required=True, help="List of model labels")
    p_compare.add_argument("--output", help="Output directory for comparison results")
    p_compare.set_defaults(func=cmd_compare)
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
