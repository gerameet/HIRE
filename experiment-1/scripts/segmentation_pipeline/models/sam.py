from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from ..core.base import PromptableSegmentationModel, ModelConfig
from ..core.data import Segment, BoundingBox
from ..core.utils import ensure_image_pil, get_device
from . import register_model


@register_model("sam")
class SAMSegmentationModel(PromptableSegmentationModel):
    """Wrapper for Meta's Segment Anything Model (SAM).
    
    Supports:
    - Automatic whole-image segmentation
    - Point-prompted segmentation
    - Box-prompted segmentation
    
    Requires: pip install segment-anything torch torchvision
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_type = config.model_type or "vit_b"
        self.use_automatic = config.extra_params.get("use_automatic", True)
        self.sam_model = None
        self.predictor = None
        self.mask_generator = None
    
    def initialize(self) -> None:
        """Initialize SAM model."""
        try:
            import torch
            from segment_anything import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor
            )
        except ImportError as e:
            raise RuntimeError(
                "SAM requires 'segment-anything' package. Install with:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from e
        
        if not self.config.checkpoint:
            raise ValueError("SAM requires checkpoint path (set config.checkpoint)")
        
        device = get_device(self.config.device)
        
        # Load model
        self.sam_model = sam_model_registry[self.model_type](
            checkpoint=self.config.checkpoint
        ).to(device)
        self.sam_model.eval()
        
        # Initialize predictor and generator
        self.predictor = SamPredictor(self.sam_model)
        
        if self.use_automatic:
            generator_params = self.config.extra_params.get("generator_params", {})
            self.mask_generator = SamAutomaticMaskGenerator(
                self.sam_model,
                **generator_params
            )
        
        self._initialized = True
        self.torch = torch
    
    def segment(
        self,
        image: Union[Image.Image, np.ndarray],
        prompts: Optional[Dict[str, Any]] = None
    ) -> List[Segment]:
        """Run SAM segmentation."""
        image = ensure_image_pil(image)
        np_img = np.array(image.convert("RGB"))
        
        # Automatic segmentation if no prompts
        if prompts is None and self.mask_generator is not None:
            return self._segment_automatic(np_img)
        
        # Prompted segmentation
        if prompts:
            return self._segment_prompted(np_img, prompts)
        
        return []
    
    def _segment_automatic(self, np_img: np.ndarray) -> List[Segment]:
        """Run automatic mask generation."""
        masks = self.mask_generator.generate(np_img)
        
        segments = []
        for m in masks:
            seg_mask = np.asarray(m["segmentation"]).astype(np.uint8)
            
            # Extract bbox (format: x, y, w, h)
            x, y, w, h = m.get("bbox", (0, 0, seg_mask.shape[1], seg_mask.shape[0]))
            bbox = BoundingBox(int(x), int(y), int(x + w), int(y + h))
            
            score = float(m.get("predicted_iou", 1.0))
            
            segments.append(Segment(
                mask=seg_mask,
                bbox=bbox,
                score=score,
                label="sam_auto",
                metadata={
                    "stability_score": m.get("stability_score"),
                    "point_coords": m.get("point_coords")
                }
            ))
        
        return segments
    
    def _segment_prompted(
        self,
        np_img: np.ndarray,
        prompts: Dict[str, Any]
    ) -> List[Segment]:
        """Run prompted segmentation."""
        self.predictor.set_image(np_img)
        
        device = get_device(self.config.device)
        
        # Parse prompts
        boxes = prompts.get("boxes")
        points = prompts.get("points")
        point_labels = prompts.get("point_labels")
        
        # Convert to tensors
        boxes_tensor = None
        if boxes:
            boxes_tensor = self.torch.tensor(boxes, device=device, dtype=self.torch.float32)
        
        points_tensor = None
        if points:
            points_tensor = self.torch.tensor(points, device=device, dtype=self.torch.float32)
        
        labels_tensor = None
        if point_labels:
            labels_tensor = self.torch.tensor(point_labels, device=device, dtype=self.torch.int32)
        
        # Predict
        with self.torch.no_grad():
            masks, scores, _ = self.predictor.predict(
                point_coords=points_tensor,
                point_labels=labels_tensor,
                box=boxes_tensor,
                multimask_output=False
            )
        
        segments = []
        for i, mask in enumerate(masks):
            seg_mask = (mask > 0.5).astype(np.uint8)
            bbox = BoundingBox.from_mask(seg_mask)
            score = float(scores[i]) if scores is not None else 1.0
            
            segments.append(Segment(
                mask=seg_mask,
                bbox=bbox,
                score=score,
                label="sam_prompt"
            ))
        
        return segments
    
    def segment_with_points(
        self,
        image: Union[Image.Image, np.ndarray],
        points: List[tuple],
        point_labels: Optional[List[int]] = None
    ) -> List[Segment]:
        """Segment with point prompts."""
        if point_labels is None:
            point_labels = [1] * len(points)  # 1 = foreground
        
        return self.segment(image, {
            "points": points,
            "point_labels": point_labels
        })
    
    def segment_with_boxes(
        self,
        image: Union[Image.Image, np.ndarray],
        boxes: List[tuple]
    ) -> List[Segment]:
        """Segment with bounding box prompts."""
        return self.segment(image, {"boxes": boxes})