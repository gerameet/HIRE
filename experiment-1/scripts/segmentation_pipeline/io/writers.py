import os
import json
from typing import List, Optional, Dict, Any
import numpy as np
from PIL import Image
import pyarrow as pa

from ..core.data import Segment, SegmentationResult
from ..core.utils import compress_mask


class ArrowWriter:
    """Write segmentation results to Arrow format."""

    @staticmethod
    def write(result: SegmentationResult, output_path: str) -> None:
        """Write result to Arrow file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Compress masks
        masks_bytes = []
        shapes = []
        for seg in result.segments:
            masks_bytes.append(compress_mask(seg.mask))
            shapes.append(seg.mask.shape)

        # Prepare data
        boxes = [seg.bbox.to_tuple() for seg in result.segments]
        scores = [seg.score for seg in result.segments]
        labels = [seg.label for seg in result.segments]
        labels = [seg.label for seg in result.segments]
        label_ids = [seg.label_id for seg in result.segments]
        areas = [seg.area for seg in result.segments]

        # Create Arrow table
        table = pa.table(
            {
                "image_path": [result.image_path],
                "image_size": [result.image_size],
                "masks": [masks_bytes],
                "shapes": [shapes],
                "boxes": [boxes],
                "scores": [scores],
                "labels": [labels],
                "label_ids": [label_ids],
                "areas": [areas],
                "processing_time": [result.processing_time],
            }
        )

        # Write to file
        with pa.OSFile(output_path, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


class CropWriter:
    """Write segmented instance crops."""

    @staticmethod
    def write(
        image: Image.Image,
        segments: List[Segment],
        output_dir: str,
        save_with_alpha: bool = True,
    ) -> None:
        """Save instance crops with optional transparency."""
        os.makedirs(output_dir, exist_ok=True)

        for idx, seg in enumerate(segments):
            x1, y1, x2, y2 = seg.bbox.to_tuple()

            # Validate and clip bbox
            w, h = image.size
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop image
            crop = image.crop((x1, y1, x2, y2))

            if save_with_alpha:
                # Apply mask as alpha channel
                crop = crop.convert("RGBA")
                mask_crop = seg.mask[y1:y2, x1:x2]
                mask_pil = Image.fromarray((mask_crop * 255).astype("uint8")).convert(
                    "L"
                )
                crop.putalpha(mask_pil)

            # Save
            filename = f"{seg.label}_{idx:04d}_{seg.score:.2f}.png"
            crop.save(os.path.join(output_dir, filename))


class VisualizationWriter:
    """Write visualization overlays."""

    @staticmethod
    def write(
        image: Image.Image,
        segments: List[Segment],
        output_path: str,
        show_boxes: bool = True,
        show_labels: bool = True,
        alpha: float = 0.5,
    ) -> None:
        """Create and save visualization with overlays."""
        import cv2

        img_array = np.array(image)
        overlay = img_array.copy()

        # Generate colors
        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(segments), 3))

        for idx, seg in enumerate(segments):
            color = tuple(map(int, colors[idx]))

            # Draw mask
            mask_colored = np.zeros_like(img_array)
            mask_colored[seg.mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1, mask_colored, alpha, 0)

            # Draw bounding box
            if show_boxes:
                x1, y1, x2, y2 = seg.bbox.to_tuple()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if show_labels:
                x1, y1, _, _ = seg.bbox.to_tuple()
                label_text = f"{seg.label}: {seg.score:.2f}"
                cv2.putText(
                    overlay,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_img = Image.fromarray(overlay)
        result_img.save(output_path)


def write_segments(
    result: SegmentationResult,
    image: Image.Image,
    output_dir: str,
    write_arrow: bool = True,
    write_crops: bool = False,
    write_viz: bool = False,
    write_original: bool = False,
) -> None:
    """Convenience function to write various outputs."""
    os.makedirs(output_dir, exist_ok=True)

    if write_arrow:
        arrow_path = os.path.join(output_dir, "segments.arrow")
        ArrowWriter.write(result, arrow_path)

    if write_crops:
        crops_dir = os.path.join(output_dir, "crops")
        CropWriter.write(image, result.segments, crops_dir)

    if write_viz:
        viz_path = os.path.join(output_dir, "visualization.png")
        VisualizationWriter.write(image, result.segments, viz_path)

    if write_original:
        orig_path = os.path.join(output_dir, "original.png")
        image.save(orig_path)
