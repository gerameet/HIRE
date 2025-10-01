from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from ..core.base import SegmentationModel, ModelConfig
from ..core.data import Segment, BoundingBox
from ..core.utils import ensure_image_pil
from . import register_model


@register_model("yolo")
class YOLOSegmentationModel(SegmentationModel):
    """Wrapper for YOLOv8 segmentation models.
    
    Supports instance segmentation with excellent speed/accuracy tradeoff.
    
    Requires: pip install ultralytics
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
    
    def initialize(self) -> None:
        """Initialize YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "YOLO requires 'ultralytics' package. Install with:\n"
                "pip install ultralytics"
            ) from e
        
        # Default to YOLOv8n-seg if no checkpoint specified
        model_name = self.config.checkpoint or "yolov8n-seg.pt"
        self.model = YOLO(model_name)
        
        # Move to device if specified
        if self.config.device:
            self.model.to(self.config.device)
        
        self._initialized = True
    
    def segment(
        self,
        image: Union[Image.Image, np.ndarray],
        prompts: Optional[Dict[str, Any]] = None
    ) -> List[Segment]:
        """Run YOLO segmentation."""
        image = ensure_image_pil(image)
        
        # Run inference
        results = self.model(
            image,
            conf=self.config.confidence_threshold,
            verbose=False
        )
        
        segments = []
        
        # Process results (YOLO returns list of Results objects)
        for result in results:
            if result.masks is None:
                continue
            
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            # Get class names
            names = result.names
            
            for mask, box, score, cls_id in zip(masks, boxes, scores, classes):
                # Resize mask to original image size
                mask_resized = Image.fromarray(mask).resize(
                    image.size,
                    Image.BILINEAR
                )
                mask_array = (np.array(mask_resized) > 0.5).astype(np.uint8)
                
                x1, y1, x2, y2 = box
                bbox = BoundingBox(
                    int(x1), int(y1), int(x2), int(y2)
                )
                
                label = names[int(cls_id)]
                
                segments.append(Segment(
                    mask=mask_array,
                    bbox=bbox,
                    score=float(score),
                    label=label,
                    label_id=int(cls_id)
                ))
        
        return segments
    
    def segment_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        prompts: Optional[List[Dict[str, Any]]] = None
    ) -> List[List[Segment]]:
        """Run batch inference (more efficient than one-by-one)."""
        images = [ensure_image_pil(img) for img in images]
        
        # Run batch inference
        results = self.model(
            images,
            conf=self.config.confidence_threshold,
            verbose=False
        )
        
        all_segments = []
        for result, image in zip(results, images):
            segments = []
            
            if result.masks is None:
                all_segments.append(segments)
                continue
            
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            names = result.names
            
            for mask, box, score, cls_id in zip(masks, boxes, scores, classes):
                mask_resized = Image.fromarray(mask).resize(
                    image.size,
                    Image.BILINEAR
                )
                mask_array = (np.array(mask_resized) > 0.5).astype(np.uint8)
                
                x1, y1, x2, y2 = box
                bbox = BoundingBox(int(x1), int(y1), int(x2), int(y2))
                
                segments.append(Segment(
                    mask=mask_array,
                    bbox=bbox,
                    score=float(score),
                    label=names[int(cls_id)],
                    label_id=int(cls_id)
                ))
            
            all_segments.append(segments)
        
        return all_segments