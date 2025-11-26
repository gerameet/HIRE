"""Adapters for integrating existing segmentation pipeline with hierarchical pipeline.

This module provides adapters to convert between the existing segmentation
pipeline's data structures and the hierarchical pipeline's data structures.
"""

from typing import List, Dict, Any
import numpy as np
import sys
from pathlib import Path

# Import from hierarchical pipeline core
from ..core.data import Part

# Add segmentation pipeline to path
seg_pipeline_path = Path(__file__).parent.parent.parent / "segmentation_pipeline"
if str(seg_pipeline_path) not in sys.path:
    sys.path.insert(0, str(seg_pipeline_path))

from segmentation_pipeline.core.data import SegmentationResult, Segment


class SegmentationAdapter:
    """Adapter to convert SegmentationResult to List[Part].

    This adapter bridges the existing segmentation pipeline with the
    hierarchical pipeline, allowing reuse of existing segmentation models.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the adapter.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.min_confidence = self.config.get("min_confidence", 0.0)
        self.min_area = self.config.get("min_area", 0)

    def convert_result(self, result: SegmentationResult) -> List[Part]:
        """Convert SegmentationResult to List[Part].

        Args:
            result: SegmentationResult from existing pipeline

        Returns:
            List of Part objects for hierarchical pipeline
        """
        parts = []

        for idx, segment in enumerate(result.segments):
            # Skip segments below thresholds
            if segment.score < self.min_confidence:
                continue
            if segment.area and segment.area < self.min_area:
                continue

            # Convert segment to part
            part = self._segment_to_part(segment, idx, result)
            parts.append(part)

        return parts

    def _segment_to_part(
        self, segment: Segment, idx: int, result: SegmentationResult
    ) -> Part:
        """Convert a single Segment to Part.

        Args:
            segment: Segment from existing pipeline
            idx: Index of segment in result
            result: Parent SegmentationResult for context

        Returns:
            Part object
        """
        # Generate unique ID
        part_id = f"part_{idx:04d}"

        # Convert mask to bool if needed
        mask = segment.mask
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)

        # Extract bbox as tuple
        bbox = segment.bbox.to_tuple()

        # Create metadata with original segment info
        metadata = {
            "label": segment.label,
            "label_id": segment.label_id,
            "area": segment.area,
            "source": "segmentation_pipeline",
            "model_info": result.model_info,
        }

        # Add any additional segment metadata
        if segment.metadata:
            metadata.update(segment.metadata)

        # Create Part
        part = Part(
            id=part_id,
            mask=mask,
            bbox=bbox,
            features=None,  # No raw features from segmentation
            embedding=None,  # Will be generated later
            confidence=segment.score,
            metadata=metadata,
        )

        return part

    def convert_segment(self, segment: Segment, segment_id: str = None) -> Part:
        """Convert a single Segment to Part (standalone).

        Useful for converting individual segments without full SegmentationResult.

        Args:
            segment: Segment to convert
            segment_id: Optional custom ID (defaults to "part_0000")

        Returns:
            Part object
        """
        part_id = segment_id or "part_0000"

        # Convert mask to bool if needed
        mask = segment.mask
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)

        # Extract bbox as tuple
        bbox = segment.bbox.to_tuple()

        # Create metadata
        metadata = {
            "label": segment.label,
            "label_id": segment.label_id,
            "area": segment.area,
            "source": "segmentation_pipeline",
        }

        if segment.metadata:
            metadata.update(segment.metadata)

        # Create Part
        part = Part(
            id=part_id,
            mask=mask,
            bbox=bbox,
            features=None,
            embedding=None,
            confidence=segment.score,
            metadata=metadata,
        )

        return part


class SegmentationDiscoveryAdapter:
    """Adapter that wraps existing segmentation models as PartDiscoveryMethod.

    This allows using existing segmentation models (YOLO, SAM, Mask2Former)
    directly in the hierarchical pipeline.
    """

    def __init__(self, segmentation_model, config: Dict[str, Any] = None):
        """Initialize the adapter.

        Args:
            segmentation_model: Instance of SegmentationModel from existing pipeline
            config: Optional configuration dictionary
        """
        self.model = segmentation_model
        self.config = config or {}
        self.adapter = SegmentationAdapter(config)

    def discover_parts(
        self, image: np.ndarray, prompts: Dict[str, Any] = None
    ) -> List[Part]:
        """Discover parts using existing segmentation model.

        Args:
            image: Input image as numpy array (H, W, C)
            prompts: Optional prompts for segmentation (e.g. text, points)

        Returns:
            List of Part objects
        """
        # Run segmentation
        segments = self.model.segment(image, prompts=prompts)

        # Convert segments to parts
        parts = []
        # Build minimal result context (handle PIL Image or numpy array)
        if hasattr(image, "size"):
            # PIL Image: size is (width, height)
            image_size = (image.size[0], image.size[1])
        else:
            # numpy array: shape (H, W, C)
            try:
                image_size = (image.shape[1], image.shape[0])
            except Exception:
                image_size = (0, 0)

        for idx, segment in enumerate(segments):
            part = self.adapter._segment_to_part(
                segment,
                idx,
                # Create minimal result context
                type(
                    "Result",
                    (),
                    {
                        "model_info": self.model.get_model_info(),
                        "image_size": image_size,
                    },
                )(),
            )
            parts.append(part)

        return parts

    def discover_parts_batch(self, images: List[np.ndarray]) -> List[List[Part]]:
        """Discover parts in a batch of images.

        Args:
            images: List of input images

        Returns:
            List of part lists (one list per image)
        """
        # Use model's batch processing if available
        batch_segments = self.model.segment_batch(images)

        # Convert all segments to parts
        all_parts = []
        for img_idx, segments in enumerate(batch_segments):
            img = images[img_idx]
            if hasattr(img, "size"):
                image_size = (img.size[0], img.size[1])
            else:
                try:
                    image_size = (img.shape[1], img.shape[0])
                except Exception:
                    image_size = (0, 0)

            parts = []
            for seg_idx, segment in enumerate(segments):
                part = self.adapter._segment_to_part(
                    segment,
                    seg_idx,
                    type(
                        "Result",
                        (),
                        {
                            "model_info": self.model.get_model_info(),
                            "image_size": image_size,
                        },
                    )(),
                )
                parts.append(part)
            all_parts.append(parts)

        return all_parts

    def get_method_info(self) -> Dict[str, Any]:
        """Get information about this discovery method.

        Returns:
            Dictionary with method metadata
        """
        return {
            "method_class": "SegmentationDiscoveryAdapter",
            "wrapped_model": self.model.get_model_info(),
            "config": self.config,
        }
