from typing import List, Dict, Any
import numpy as np
import pyarrow as pa

from ..core.data import Segment, BoundingBox
from ..core.utils import decompress_mask


def segments_to_coco(
    segments: List[Segment],
    image_id: int,
    image_info: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Convert segments to COCO format annotations."""
    annotations = []
    
    for idx, seg in enumerate(segments):
        # Convert mask to RLE or polygon (simplified here)
        mask_rle = _mask_to_rle(seg.mask)
        
        ann = {
            "id": idx,
            "image_id": image_id,
            "category_id": seg.label_id or 0,
            "bbox": seg.bbox.to_xywh(),
            "area": seg.area,
            "segmentation": mask_rle,
            "score": seg.score,
            "iscrowd": 0
        }
        annotations.append(ann)
    
    return annotations


def segments_to_yolo(segments: List[Segment], img_width: int, img_height: int) -> List[str]:
    """Convert segments to YOLO format (normalized polygons)."""
    yolo_lines = []
    
    for seg in segments:
        class_id = seg.label_id or 0
        
        # Extract contours from mask (simplified)
        contours = _mask_to_polygon(seg.mask)
        
        if not contours:
            continue
        
        # Normalize coordinates
        normalized = []
        for x, y in contours:
            normalized.extend([x / img_width, y / img_height])
        
        # Format: class_id x1 y1 x2 y2 ...
        line = f"{class_id} " + " ".join(f"{c:.6f}" for c in normalized)
        yolo_lines.append(line)
    
    return yolo_lines


def load_from_arrow(arrow_path: str) -> Dict[str, Any]:
    """Load segmentation data from Arrow file."""
    table = pa.ipc.open_file(arrow_path).read_all()
    
    # Convert to Python objects
    row = table.to_pydict()
    
    # Decompress masks
    masks_bytes = row["masks"][0]
    shapes = row["shapes"][0]
    masks = [decompress_mask(mb, shape) for mb, shape in zip(masks_bytes, shapes)]
    
    segments = []
    for i in range(len(masks)):
        seg = Segment(
            mask=masks[i],
            bbox=BoundingBox(*row["boxes"][0][i]),
            score=row["scores"][0][i],
            label=row["labels"][0][i],
            label_id=row["label_ids"][0][i],
            area=row["areas"][0][i]
        )
        segments.append(seg)
    
    return {
        "image_path": row["image_path"][0],
        "image_size": tuple(row["image_size"][0]),
        "segments": segments,
        "processing_time": row["processing_time"][0]
    }


def _mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Convert binary mask to RLE (simplified)."""
    # This is a placeholder - use pycocotools for production
    return {"counts": [], "size": list(mask.shape)}


def _mask_to_polygon(mask: np.ndarray) -> List[tuple]:
    """Convert mask to polygon contour (simplified)."""
    import cv2
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    # Return largest contour
    contour = max(contours, key=cv2.contourArea)
    return [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
