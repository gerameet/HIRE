#!/usr/bin/env python3
"""Run Phase 1 pipeline: segmentation -> parts -> bottom-up hierarchy -> visualization

Usage examples:
  python run_phase1.py --images data/images --model dummy --output output/phase1

This script is intentionally minimal and uses the existing segmentation
pipeline's model registry. It writes parse graphs (JSON) and mask overlays.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List


def add_scripts_to_path():
    # Ensure experiment-1/scripts is on sys.path so package imports work
    repo_root = Path(__file__).resolve().parents[1]
    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)


def parse_args():
    p = argparse.ArgumentParser(description="Run Phase 1 hierarchical pipeline")
    p.add_argument(
        "--images", required=True, help="Image file or directory containing images"
    )
    p.add_argument(
        "--model", default="dummy", help="Segmentation model name (registered)"
    )
    p.add_argument(
        "--output", default="experiment-1/output/phase1", help="Output directory"
    )
    p.add_argument("--device", default=None, help="Device string, e.g. cuda:0 or cpu")
    p.add_argument(
        "--save-overlay", action="store_true", help="Save mask overlay images"
    )
    p.add_argument("--save-graph", action="store_true", help="Save parse graph JSON")
    return p.parse_args()


def list_images(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        return [str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in exts]
    raise ValueError(f"Images path not found: {path}")


def main():
    args = parse_args()
    add_scripts_to_path()

    # Import local packages after adjusting sys.path
    from hierarchical_pipeline.config import (
        load_config,
        create_default_config,
        validate_config,
    )
    from hierarchical_pipeline.utils.gpu import GPUManager
    from hierarchical_pipeline.adapters.segmentation import SegmentationDiscoveryAdapter
    from hierarchical_pipeline.core.builder import BottomUpHierarchyBuilder
    from hierarchical_pipeline.visualization import overlay_masks

    # Import segmentation model registry
    from segmentation_pipeline.models import get_model, ModelConfig

    images = list_images(args.images)
    os.makedirs(args.output, exist_ok=True)

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
    model_cfg = ModelConfig(device=str(gpu_mgr.device) if gpu_mgr.device else None)
    try:
        seg_model = get_model(args.model, model_cfg)
    except Exception as e:
        print(f"Failed to create segmentation model '{args.model}': {e}")
        return

    # Initialize model (context manager supported)
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

        for img_path in images:
            print(f"Processing {img_path}")
            img = Image.open(img_path).convert("RGB")

            parts = adapter.discover_parts(img)

            # attach image metadata to parts
            for p in parts:
                p.metadata.setdefault("image_path", img_path)
                p.metadata.setdefault("image_size", (img.width, img.height))

            graph = builder.build_hierarchy(parts)

            stem = Path(img_path).stem
            out_prefix = Path(args.output) / stem

            if args.save_graph:
                json_path = str(out_prefix.with_suffix(".graph.json"))
                with open(json_path, "w") as f:
                    f.write(graph.to_json())
                print(f"Saved graph: {json_path}")

            if args.save_overlay:
                # Save overlay of all masks on image
                masks = [p.mask for p in parts]
                if masks:
                    import numpy as np

                    img_np = np.array(img)
                    overlay = overlay_masks(img_np, masks)
                    from PIL import Image as PILImage

                    overlay_pil = PILImage.fromarray(overlay)
                    overlay_path = str(out_prefix.with_suffix(".overlay.png"))
                    overlay_pil.save(overlay_path)
                    print(f"Saved overlay: {overlay_path}")

    print("Phase 1 run complete")


if __name__ == "__main__":
    main()
