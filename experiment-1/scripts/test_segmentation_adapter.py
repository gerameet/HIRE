"""Test script for segmentation adapter.

Tests the adapter with SAM model to verify:
1. SegmentationResult → List[Part] conversion works
2. GPU usage is properly detected
3. Parts have correct structure
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

from segmentation_pipeline.core.base import ModelConfig
from segmentation_pipeline.core.data import SegmentationResult
from hierarchical_pipeline.adapters import (
    SegmentationAdapter,
    SegmentationDiscoveryAdapter,
)


def test_adapter_basic():
    """Test basic adapter functionality with mock data."""
    print("\n=== Test 1: Basic Adapter Functionality ===")

    from segmentation_pipeline.core.data import Segment, BoundingBox

    # Create mock segmentation result
    segments = [
        Segment(
            mask=np.ones((100, 100), dtype=np.uint8),
            bbox=BoundingBox(10, 10, 90, 90),
            score=0.95,
            label="object1",
            label_id=1,
        ),
        Segment(
            mask=np.ones((50, 50), dtype=np.uint8),
            bbox=BoundingBox(20, 20, 70, 70),
            score=0.85,
            label="object2",
            label_id=2,
        ),
    ]

    result = SegmentationResult(
        image_path="test.jpg",
        segments=segments,
        image_size=(100, 100),
        model_info={"model": "test"},
    )

    # Convert using adapter
    adapter = SegmentationAdapter()
    parts = adapter.convert_result(result)

    # Verify conversion
    assert len(parts) == 2, f"Expected 2 parts, got {len(parts)}"

    for i, part in enumerate(parts):
        print(f"\nPart {i}:")
        print(f"  ID: {part.id}")
        print(f"  Mask shape: {part.mask.shape}")
        print(f"  Mask dtype: {part.mask.dtype}")
        print(f"  BBox: {part.bbox}")
        print(f"  Confidence: {part.confidence}")
        print(f"  Area: {part.get_area()}")
        print(f"  Metadata: {part.metadata}")

        # Verify structure
        assert part.mask.ndim == 2, "Mask should be 2D"
        assert part.mask.dtype in [np.bool_, np.uint8], "Mask should be bool or uint8"
        assert len(part.bbox) == 4, "BBox should have 4 elements"
        assert part.confidence > 0, "Confidence should be positive"
        assert "label" in part.metadata, "Metadata should contain label"

    print("\n✓ Basic adapter test passed!")
    return True


def test_adapter_with_sam():
    """Test adapter with real SAM model."""
    print("\n=== Test 2: Adapter with SAM Model ===")

    try:
        import torch
        from segmentation_pipeline.models.sam import SAMSegmentationModel
    except ImportError as e:
        print(f"⚠ Skipping SAM test: {e}")
        return False

    # Check for SAM checkpoint
    sam_checkpoint = (
        Path(__file__).parent
        / "hierarchical_pipeline"
        / "models"
        / "sam_vit_b_01ec64.pth"
    )
    if not sam_checkpoint.exists():
        print(f"⚠ Skipping SAM test: Checkpoint not found at {sam_checkpoint}")
        print(
            "  Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )
        return False

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Load test image
    test_image_path = Path(__file__).parent.parent / "images" / "image.png"
    if not test_image_path.exists():
        print(f"⚠ Test image not found at {test_image_path}")
        return False

    image = Image.open(test_image_path).convert("RGB")
    image_np = np.array(image)
    print(f"Image shape: {image_np.shape}")

    # Initialize SAM model
    print("\nInitializing SAM model...")
    config = ModelConfig(
        device=device,
        checkpoint=str(sam_checkpoint),
        model_type="vit_b",
        extra_params={
            "use_automatic": True,
            "generator_params": {
                "points_per_side": 16,  # Reduced for faster testing
                "pred_iou_thresh": 0.8,
                "stability_score_thresh": 0.85,
            },
        },
    )

    sam_model = SAMSegmentationModel(config)
    sam_model.initialize()

    # Run segmentation
    print("Running SAM segmentation...")
    start_time = time.time()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    segments = sam_model.segment(image_np)

    seg_time = time.time() - start_time
    print(f"Segmentation time: {seg_time:.2f}s")
    print(f"Number of segments: {len(segments)}")

    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {peak_memory:.2f} GB")

    # Create SegmentationResult
    result = SegmentationResult(
        image_path=str(test_image_path),
        segments=segments,
        image_size=(image_np.shape[1], image_np.shape[0]),
        processing_time=seg_time,
        model_info=sam_model.get_model_info(),
    )

    # Convert to Parts using adapter
    print("\nConverting to Parts...")
    adapter = SegmentationAdapter(config={"min_confidence": 0.8, "min_area": 100})
    parts = adapter.convert_result(result)

    print(f"Number of parts after filtering: {len(parts)}")

    # Verify parts
    if len(parts) > 0:
        print("\nSample parts:")
        for i, part in enumerate(parts[:3]):  # Show first 3
            print(f"\nPart {i}:")
            print(f"  ID: {part.id}")
            print(f"  Mask shape: {part.mask.shape}")
            print(f"  Mask dtype: {part.mask.dtype}")
            print(f"  BBox: {part.bbox}")
            print(f"  Confidence: {part.confidence:.3f}")
            print(f"  Area: {part.get_area()}")
            print(f"  Label: {part.metadata.get('label')}")

        # Verify all parts have correct structure
        for part in parts:
            assert part.mask.shape == (
                image_np.shape[0],
                image_np.shape[1],
            ), f"Mask shape mismatch: {part.mask.shape} vs {image_np.shape[:2]}"
            assert part.mask.dtype in [
                np.bool_,
                np.uint8,
            ], f"Invalid mask dtype: {part.mask.dtype}"
            assert len(part.bbox) == 4, "BBox should have 4 elements"
            assert 0 <= part.confidence <= 1, f"Invalid confidence: {part.confidence}"

    print("\n✓ SAM adapter test passed!")
    return True


def test_discovery_adapter():
    """Test SegmentationDiscoveryAdapter wrapper with dummy model."""
    print("\n=== Test 3: Discovery Adapter Wrapper ===")

    try:
        from segmentation_pipeline.models.dummy import DummySegmentationModel
    except ImportError as e:
        print(f"⚠ Skipping discovery adapter test: {e}")
        return False

    # Load test image
    test_image_path = Path(__file__).parent.parent / "images" / "image.png"
    if not test_image_path.exists():
        print(f"⚠ Test image not found at {test_image_path}")
        return False

    image = Image.open(test_image_path).convert("RGB")
    image_np = np.array(image)
    print(f"Image shape: {image_np.shape}")

    # Initialize dummy model (no GPU needed)
    config = ModelConfig(extra_params={"num_masks": 5})

    dummy_model = DummySegmentationModel(config)
    dummy_model.initialize()

    # Wrap with discovery adapter
    print("Creating discovery adapter...")
    discovery = SegmentationDiscoveryAdapter(
        dummy_model, config={"min_confidence": 0.5, "min_area": 100}
    )

    # Test discover_parts method
    print("Running discover_parts...")
    parts = discovery.discover_parts(image_np)

    print(f"Discovered {len(parts)} parts")

    # Verify parts
    assert len(parts) > 0, "Should discover at least one part"

    for i, part in enumerate(parts[:3]):  # Show first 3
        print(f"\nPart {i}:")
        print(f"  ID: {part.id}")
        print(f"  Mask shape: {part.mask.shape}")
        print(f"  BBox: {part.bbox}")
        print(f"  Confidence: {part.confidence:.3f}")
        print(f"  Area: {part.get_area()}")

    for part in parts:
        assert part.mask.shape == (
            image_np.shape[0],
            image_np.shape[1],
        ), f"Mask shape mismatch: {part.mask.shape} vs {image_np.shape[:2]}"
        assert part.mask.dtype in [
            np.bool_,
            np.uint8,
        ], f"Invalid mask dtype: {part.mask.dtype}"
        assert len(part.bbox) == 4, "BBox should have 4 elements"

    # Test method info
    info = discovery.get_method_info()
    print(f"\nMethod info: {info}")
    assert "wrapped_model" in info
    assert info["method_class"] == "SegmentationDiscoveryAdapter"

    print("\n✓ Discovery adapter test passed!")
    return True


def test_gpu_detection():
    """Test GPU detection and usage reporting."""
    print("\n=== Test 4: GPU Detection ===")

    try:
        import torch
    except ImportError:
        print("⚠ PyTorch not available, skipping GPU test")
        return False

    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {gpu_available}")

    if gpu_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(f"CUDA Version: {torch.version.cuda}")

        # Test memory allocation
        try:
            test_tensor = torch.randn(1000, 1000, device="cuda")
            print(f"✓ Successfully allocated tensor on GPU")
            print(f"  Current GPU memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠ GPU allocation test failed: {e}")
            return False
    else:
        print("No GPU available - will use CPU for processing")

    print("\n✓ GPU detection test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Segmentation Adapter")
    print("=" * 60)

    results = []

    # Test 1: Basic functionality
    try:
        results.append(("Basic Adapter", test_adapter_basic()))
    except Exception as e:
        print(f"\n✗ Basic adapter test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Basic Adapter", False))

    # Test 2: SAM integration (optional - requires checkpoint)
    try:
        results.append(("SAM Adapter", test_adapter_with_sam()))
    except Exception as e:
        print(f"\n✗ SAM adapter test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("SAM Adapter", False))

    # Test 3: Discovery adapter (required)
    try:
        results.append(("Discovery Adapter", test_discovery_adapter()))
    except Exception as e:
        print(f"\n✗ Discovery adapter test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Discovery Adapter", False))

    # Test 4: GPU detection (informational)
    try:
        results.append(("GPU Detection", test_gpu_detection()))
    except Exception as e:
        print(f"\n✗ GPU detection test failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(("GPU Detection", False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED/SKIPPED"
        print(f"{name}: {status}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    # Consider test successful if core tests pass (Basic + Discovery)
    core_tests_passed = results[0][1] and results[2][1]  # Basic and Discovery

    if core_tests_passed:
        print("\n✓ Core adapter functionality verified!")
    else:
        print("\n✗ Core adapter tests failed!")

    return core_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
