from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from ..core.base import SegmentationModel, ModelConfig
from ..core.data import Segment, BoundingBox
from ..core.utils import ensure_image_pil
from . import register_model


@register_model("dummy")
class DummySegmentationModel(SegmentationModel):
    """Lightweight dummy model for testing the pipeline.
    
    Produces toy circular masks using deterministic randomness.
    No dependencies required beyond numpy.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.num_masks = config.extra_params.get("num_masks", 3)
    
    def initialize(self) -> None:
        """No initialization needed for dummy model."""
        self._initialized = True
    
    def segment(
        self,
        image: Union[Image.Image, np.ndarray],
        prompts: Optional[Dict[str, Any]] = None
    ) -> List[Segment]:
        """Generate dummy circular masks."""
        image = ensure_image_pil(image)
        w, h = image.size
        cx, cy = w // 2, h // 2
        
        segments = []
        rng = np.random.RandomState(seed=(w + h))
        
        for i in range(self.num_masks):
            # Generate random circle
            r = int(min(w, h) * (0.15 + 0.15 * rng.rand()))
            ox = int((rng.rand() - 0.5) * w * 0.4)
            oy = int((rng.rand() - 0.5) * h * 0.4)
            center_x = np.clip(cx + ox, 0, w - 1)
            center_y = np.clip(cy + oy, 0, h - 1)
            
            # Create circular mask
            Y, X = np.ogrid[:h, :w]
            mask = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= (r * r)
            mask = mask.astype(np.uint8)
            
            # Create bounding box
            x1 = int(max(0, center_x - r))
            y1 = int(max(0, center_y - r))
            x2 = int(min(w, center_x + r))
            y2 = int(min(h, center_y + r))
            bbox = BoundingBox(x1, y1, x2, y2)
            
            # Random score
            score = float(0.5 + 0.5 * rng.rand())
            
            segments.append(Segment(
                mask=mask,
                bbox=bbox,
                score=score,
                label=f"dummy_{i}",
                label_id=i
            ))
        
        return segments
    
    def uses_gpu(self) -> bool:
        return False
