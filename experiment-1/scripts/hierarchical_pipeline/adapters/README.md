# Segmentation Pipeline Adapters

This module provides adapters to integrate the existing segmentation pipeline with the hierarchical visual pipeline.

## Overview

The adapters allow you to:
1. Convert `SegmentationResult` objects to `List[Part]` for hierarchical processing
2. Use existing segmentation models (YOLO, SAM, Mask2Former) as `PartDiscoveryMethod` implementations
3. Maintain compatibility with the existing segmentation infrastructure

## Components

### SegmentationAdapter

Converts segmentation results to hierarchical pipeline format.

```python
from hierarchical_pipeline.adapters import SegmentationAdapter
from segmentation_pipeline.core.data import SegmentationResult

# Create adapter with optional filtering
adapter = SegmentationAdapter(config={
    "min_confidence": 0.8,  # Filter segments below this confidence
    "min_area": 100,        # Filter segments smaller than this
})

# Convert result
parts = adapter.convert_result(segmentation_result)
```

### SegmentationDiscoveryAdapter

Wraps existing segmentation models to implement the `PartDiscoveryMethod` interface.

```python
from hierarchical_pipeline.adapters import SegmentationDiscoveryAdapter
from segmentation_pipeline.models.sam import SAMSegmentationModel
from segmentation_pipeline.core.base import ModelConfig

# Initialize segmentation model
config = ModelConfig(
    device="cuda",
    checkpoint="models/sam_vit_b_01ec64.pth",
    model_type="vit_b",
    extra_params={"use_automatic": True}
)
sam_model = SAMSegmentationModel(config)
sam_model.initialize()

# Wrap with discovery adapter
discovery = SegmentationDiscoveryAdapter(
    sam_model,
    config={"min_confidence": 0.85, "min_area": 500}
)

# Use in hierarchical pipeline
parts = discovery.discover_parts(image_np)
```

## Data Conversion

### Segment → Part Mapping

| Segment Field | Part Field | Notes |
|--------------|------------|-------|
| `mask` | `mask` | Converted to bool dtype |
| `bbox` | `bbox` | Converted to tuple (x1, y1, x2, y2) |
| `score` | `confidence` | Direct mapping |
| `label` | `metadata["label"]` | Stored in metadata |
| `label_id` | `metadata["label_id"]` | Stored in metadata |
| `area` | `metadata["area"]` | Stored in metadata |
| `metadata` | `metadata` | Merged into part metadata |

### Additional Metadata

The adapter adds:
- `source`: Always set to `"segmentation_pipeline"`
- `model_info`: Model information from the segmentation result

## Usage Examples

### Example 1: Convert Existing Results

```python
from hierarchical_pipeline.adapters import SegmentationAdapter

# You have a SegmentationResult from existing pipeline
result = run_segmentation(image_path, model="sam")

# Convert to Parts
adapter = SegmentationAdapter()
parts = adapter.convert_result(result)

# Now use in hierarchical pipeline
for part in parts:
    print(f"Part {part.id}: area={part.get_area()}, conf={part.confidence}")
```

### Example 2: Use SAM as Part Discovery

```python
from hierarchical_pipeline.adapters import SegmentationDiscoveryAdapter
from segmentation_pipeline.models.sam import SAMSegmentationModel
from segmentation_pipeline.core.base import ModelConfig
import numpy as np
from PIL import Image

# Load image
image = np.array(Image.open("test.jpg").convert("RGB"))

# Initialize SAM
config = ModelConfig(
    device="cuda",
    checkpoint="models/sam_vit_b_01ec64.pth",
    model_type="vit_b",
    extra_params={
        "use_automatic": True,
        "generator_params": {
            "points_per_side": 32,
            "pred_iou_thresh": 0.86,
        }
    }
)
sam = SAMSegmentationModel(config)
sam.initialize()

# Wrap with adapter
discovery = SegmentationDiscoveryAdapter(sam, config={
    "min_confidence": 0.85,
    "min_area": 500,
})

# Discover parts
parts = discovery.discover_parts(image)
print(f"Discovered {len(parts)} parts")
```

### Example 3: Batch Processing

```python
# Process multiple images
images = [np.array(Image.open(p)) for p in image_paths]

# Batch discovery
all_parts = discovery.discover_parts_batch(images)

for i, parts in enumerate(all_parts):
    print(f"Image {i}: {len(parts)} parts")
```

## Configuration Options

### SegmentationAdapter Config

```python
config = {
    "min_confidence": 0.0,  # Minimum confidence threshold (0-1)
    "min_area": 0,          # Minimum area in pixels
}
```

### SegmentationDiscoveryAdapter Config

Inherits from `SegmentationAdapter` config, plus:
- All configuration is passed to the underlying `SegmentationAdapter`
- The wrapped model's configuration is separate

## GPU Usage

The adapters automatically detect and use GPU when available:

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Using CPU")

# Configure model for GPU
config = ModelConfig(device=device, ...)
```

## Integration with Hierarchical Pipeline

The adapters are designed to work seamlessly with the hierarchical pipeline:

```python
from hierarchical_pipeline import (
    SegmentationDiscoveryAdapter,
    EmbeddingMethod,
    HierarchyBuilder,
)

# 1. Part Discovery (using adapter)
discovery = SegmentationDiscoveryAdapter(sam_model)
parts = discovery.discover_parts(image)

# 2. Embedding Generation
embedding_method = DINOEmbeddingMethod(config)
for part in parts:
    part.embedding = embedding_method.embed_part(image, part.mask)

# 3. Hierarchy Construction
hierarchy_builder = BottomUpHierarchyBuilder(config)
parse_graph = hierarchy_builder.build_hierarchy(parts)

# 4. Analysis
print(f"Parse graph depth: {parse_graph.get_depth()}")
print(f"Total nodes: {len(parse_graph.nodes)}")
```

## Requirements

- Python 3.8+
- NumPy
- Existing segmentation pipeline
- PyTorch (for GPU support)
- segment-anything (for SAM)

## Testing

Run the test suite:

```bash
cd experiment-1/scripts
python test_segmentation_adapter.py
```

Run the example:

```bash
python example_sam_adapter.py
```

## Performance Notes

### GPU Acceleration

- SAM benefits significantly from GPU acceleration
- Expect 5-10x speedup on GPU vs CPU
- GPU memory usage depends on image size and model

### Filtering

- Apply filtering at adapter level for efficiency
- `min_confidence` and `min_area` reduce downstream processing
- Adjust thresholds based on your use case

### Batch Processing

- Use `discover_parts_batch()` for multiple images
- Batch processing is more efficient on GPU
- Memory usage scales with batch size

## Troubleshooting

### Import Errors

If you get import errors, ensure paths are set correctly:

```python
import sys
from pathlib import Path

# Add segmentation pipeline to path
seg_path = Path(__file__).parent.parent / "segmentation_pipeline"
sys.path.insert(0, str(seg_path))

# Add hierarchical pipeline to path
hier_path = Path(__file__).parent.parent / "hierarchical_pipeline"
sys.path.insert(0, str(hier_path))
```

### GPU Out of Memory

If you run out of GPU memory:
1. Reduce `points_per_side` in SAM config
2. Process images in smaller batches
3. Reduce image resolution
4. Use CPU fallback

### No Parts Discovered

If no parts are discovered:
1. Lower `min_confidence` threshold
2. Lower `min_area` threshold
3. Check SAM generator parameters
4. Verify image is valid RGB

## Future Enhancements

Potential improvements:
- [ ] Support for prompted segmentation (points, boxes, text)
- [ ] Caching of segmentation results
- [ ] Multi-scale part discovery
- [ ] Integration with other segmentation models
- [ ] Parallel processing for large datasets
