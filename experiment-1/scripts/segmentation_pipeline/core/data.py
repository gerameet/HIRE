from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box in xyxy format."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    def area(self) -> int:
        """Calculate bounding box area."""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Convert to xywh format."""
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)
    
    @classmethod
    def from_mask(cls, mask: np.ndarray) -> BoundingBox:
        """Create bounding box from mask."""
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return cls(0, 0, 0, 0)
        return cls(
            x1=int(xs.min()),
            y1=int(ys.min()),
            x2=int(xs.max()) + 1,
            y2=int(ys.max()) + 1
        )


@dataclass
class Segment:
    """Single segmentation instance."""
    mask: np.ndarray  # Binary mask (H, W) with dtype uint8 or bool
    bbox: BoundingBox
    score: float
    label: str = "unknown"
    label_id: Optional[int] = None
    area: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize data."""
        # Ensure mask is 2D
        if self.mask.ndim != 2:
            self.mask = np.squeeze(self.mask)
        if self.mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape {self.mask.shape}")
        
        # Ensure mask is uint8
        if self.mask.dtype != np.uint8:
            self.mask = (self.mask > 0).astype(np.uint8)
        
        # Calculate area if not provided
        if self.area is None:
            self.area = int(self.mask.sum())
        
        # Convert bbox tuple to BoundingBox if needed
        if not isinstance(self.bbox, BoundingBox):
            if isinstance(self.bbox, (tuple, list)):
                self.bbox = BoundingBox(*self.bbox)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return {
            "bbox": self.bbox.to_tuple(),
            "score": float(self.score),
            "label": self.label,
            "label_id": self.label_id,
            "area": self.area,
            "metadata": self.metadata,
        }
    
@dataclass
class SegmentationResult:
    """Complete segmentation result for one image."""
    image_path: str
    segments: List[Segment]
    image_size: Tuple[int, int]  # (width, height)
    processing_time: Optional[float] = None
    model_info: Dict[str, Any] = field(default_factory=dict)
    
    def filter_by_area(self, min_area: int, max_area: Optional[int] = None) -> "SegmentationResult":
        """Filter segments by area."""
        filtered = [
            s for s in self.segments
            if s.area >= min_area and (max_area is None or s.area <= max_area)
        ]
        return SegmentationResult(
            image_path=self.image_path,
            segments=filtered,
            image_size=self.image_size,
            processing_time=self.processing_time,
            model_info=self.model_info
        )
    
    def filter_by_score(self, min_score: float) -> "SegmentationResult":
        """Filter segments by confidence score."""
        filtered = [s for s in self.segments if s.score >= min_score]
        return SegmentationResult(
            image_path=self.image_path,
            segments=filtered,
            image_size=self.image_size,
            processing_time=self.processing_time,
            model_info=self.model_info
        )
    
    def filter_by_label(self, labels: List[str]) -> "SegmentationResult":
        """Filter segments by label."""
        filtered = [s for s in self.segments if s.label in labels]
        return SegmentationResult(
            image_path=self.image_path,
            segments=filtered,
            image_size=self.image_size,
            processing_time=self.processing_time,
            model_info=self.model_info
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.segments:
            return {
                "num_segments": 0,
                "labels": [],
                "total_area": 0,
                "avg_score": 0.0
            }
        
        label_counts = {}
        for seg in self.segments:
            label_counts[seg.label] = label_counts.get(seg.label, 0) + 1
        
        return {
            "num_segments": len(self.segments),
            "labels": label_counts,
            "total_area": sum(s.area for s in self.segments),
            "avg_score": np.mean([s.score for s in self.segments]),
            "min_score": min(s.score for s in self.segments),
            "max_score": max(s.score for s in self.segments),
        }