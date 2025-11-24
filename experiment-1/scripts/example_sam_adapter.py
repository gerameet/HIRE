"""Example: Using SAM with Hierarchical Pipeline via Adapter.

This script demonstrates how to:
1. Load an existing SAM segmentation model
2. Use the adapter to convert segments to Parts
3. Verify GPU usage
4. Process images through the hierarchical pipeline

Requirements:
- SAM checkpoint downloaded (see README)
- Test images available
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import time

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "segmentation_pipeline"))
sys.path.insert(0, str(project_root / "hierarchical_pipeline"))


def main():
    """Run SAM adapter example."""
    print("=" * 70)
    print("SAM Adapter Example - Hierarchical Visual Pipeline")
    print("=" * 70)

    # Check dependencies
    try:
        import torch
        from segmentation_pipeline.core.base import ModelConfig
        from segmentation_pipeline.models.sam import SAMSegmentationModel
        from hierarchical_pipeline.adapters import SegmentationDiscoveryAdapter
        from hierarchical_pipeline.utils.gpu import get_device
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("\nInstall required packages:")
        print("  pip install torch torchvision")
        print(
            "  pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
        return 1

    # Check GPU
    device = get_device()
    print(f"\n1. Device Configuration")
    print(f"   Using device: {device}")

    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("   Note: Running on CPU (slower)")

    # Check for SAM checkpoint
    print(f"\n2. Loading SAM Model")
    sam_checkpoint = Path("models/sam_vit_b_01ec64.pth")

    if not sam_checkpoint.exists():
        print(f"   ✗ SAM checkpoint not found at: {sam_checkpoint}")
        print(f"\n   Download SAM checkpoint:")
        print(f"   mkdir -p models")
        print(
            f"   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/"
        )
        return 1

    print(f"   ✓ Found checkpoint: {sam_checkpoint}")

    # Initialize SAM
    config = ModelConfig(
        device=device,
        checkpoint=str(sam_checkpoint),
        model_type="vit_b",
        extra_params={
            "use_automatic": True,
            "generator_params": {
                "points_per_side": 32,  # Higher = more masks, slower
                "pred_iou_thresh": 0.86,
                "stability_score_thresh": 0.92,
                "crop_n_layers": 1,
                "crop_n_points_downscale_factor": 2,
            },
        },
    )

    print(f"   Initializing SAM model...")
    sam_model = SAMSegmentationModel(config)
    sam_model.initialize()
    print(f"   ✓ SAM model ready")

    # Wrap with discovery adapter
    print(f"\n3. Creating Hierarchical Pipeline Adapter")
    discovery = SegmentationDiscoveryAdapter(
        sam_model,
        config={
            "min_confidence": 0.85,  # Filter low-quality masks
            "min_area": 500,  # Filter small masks
        },
    )
    print(f"   ✓ Adapter created")

    # Load test image
    print(f"\n4. Loading Test Image")
    test_image_path = Path(__file__).parent.parent / "images" / "image.png"

    if not test_image_path.exists():
        print(f"   ✗ Test image not found at: {test_image_path}")
        return 1

    image = Image.open(test_image_path).convert("RGB")
    image_np = np.array(image)
    print(f"   ✓ Loaded image: {test_image_path.name}")
    print(
        f"   Image size: {image_np.shape[1]}x{image_np.shape[0]} ({image_np.shape[2]} channels)"
    )

    # Run part discovery
    print(f"\n5. Running Part Discovery")
    print(f"   Processing with SAM...")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    parts = discovery.discover_parts(image_np)
    elapsed = time.time() - start_time

    print(f"   ✓ Discovery complete in {elapsed:.2f}s")
    print(f"   Discovered {len(parts)} parts (after filtering)")

    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"   Peak GPU memory: {peak_memory:.2f} GB")

    # Analyze parts
    print(f"\n6. Part Analysis")

    if len(parts) == 0:
        print(f"   No parts discovered (try lowering thresholds)")
        return 0

    # Statistics
    areas = [part.get_area() for part in parts]
    confidences = [part.confidence for part in parts]

    print(f"   Total parts: {len(parts)}")
    print(f"   Area range: {min(areas)} - {max(areas)} pixels")
    print(f"   Mean area: {np.mean(areas):.0f} pixels")
    print(f"   Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
    print(f"   Mean confidence: {np.mean(confidences):.3f}")

    # Show sample parts
    print(f"\n7. Sample Parts (first 5)")
    for i, part in enumerate(parts[:5]):
        print(f"\n   Part {i}:")
        print(f"     ID: {part.id}")
        print(f"     BBox: {part.bbox}")
        print(f"     Area: {part.get_area()} pixels")
        print(f"     Confidence: {part.confidence:.3f}")
        print(f"     Mask shape: {part.mask.shape}")
        print(f"     Mask dtype: {part.mask.dtype}")
        print(f"     Label: {part.metadata.get('label', 'N/A')}")

    # Verify structure
    print(f"\n8. Verification")
    all_valid = True

    for part in parts:
        # Check mask
        if part.mask.shape != (image_np.shape[0], image_np.shape[1]):
            print(f"   ✗ Part {part.id}: Invalid mask shape")
            all_valid = False

        if part.mask.dtype not in [np.bool_, np.uint8]:
            print(f"   ✗ Part {part.id}: Invalid mask dtype")
            all_valid = False

        # Check bbox
        if len(part.bbox) != 4:
            print(f"   ✗ Part {part.id}: Invalid bbox")
            all_valid = False

        # Check confidence
        if not (0 <= part.confidence <= 1):
            print(f"   ✗ Part {part.id}: Invalid confidence")
            all_valid = False

    if all_valid:
        print(f"   ✓ All parts have valid structure")

    # Next steps
    print(f"\n9. Next Steps")
    print(f"   ✓ Parts are ready for hierarchical pipeline")
    print(f"   - Generate embeddings (DINO, CLIP, etc.)")
    print(f"   - Build parse graph from spatial relationships")
    print(f"   - Integrate with knowledge graphs")
    print(f"   - Visualize hierarchical structure")

    print(f"\n" + "=" * 70)
    print(f"✓ Example completed successfully!")
    print(f"=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
