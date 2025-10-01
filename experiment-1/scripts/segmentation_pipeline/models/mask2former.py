from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from ..core.base import SegmentationModel, ModelConfig
from ..core.data import Segment, BoundingBox
from ..core.utils import ensure_image_pil, get_device
from . import register_model


@register_model("mask2former")
class Mask2FormerSegmentationModel(SegmentationModel):
    """Wrapper for Hugging Face Mask2Former models.
    
    Supports:
    - Instance segmentation
    - Semantic segmentation
    - Panoptic segmentation
    
    Requires: pip install transformers torch
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.task = config.extra_params.get("task", "instance")
        if self.task not in ["instance", "semantic", "panoptic"]:
            raise ValueError(f"Invalid task '{self.task}'. Must be instance/semantic/panoptic")
        self.processor = None
        self.model = None
    
    def initialize(self) -> None:
        """Initialize Mask2Former model."""
        try:
            import torch
            from transformers import (
                Mask2FormerForUniversalSegmentation,
                Mask2FormerImageProcessor
            )
        except ImportError as e:
            raise RuntimeError(
                "Mask2Former requires 'transformers' and 'torch'. Install with:\n"
                "pip install transformers torch"
            ) from e
        
        checkpoint = self.config.checkpoint or "facebook/mask2former-swin-base-coco-instance"
        device = get_device(self.config.device)
        
        self.processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)
        self.model.to(device)
        self.model.eval()
        
        self._initialized = True
        self.torch = torch
    
    def segment(
        self,
        image: Union[Image.Image, np.ndarray],
        prompts: Optional[Dict[str, Any]] = None
    ) -> List[Segment]:
        """Run Mask2Former segmentation."""
        image = ensure_image_pil(image).convert("RGB")
        
        # Prepare inputs
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(get_device(self.config.device)) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        target_size = [image.size[::-1]]  # (H, W)
        
        # Post-process based on task
        if self.task == "instance":
            return self._post_process_instance(outputs, target_size)
        elif self.task == "semantic":
            return self._post_process_semantic(outputs, target_size)
        elif self.task == "panoptic":
            return self._post_process_panoptic(outputs, target_size)
    
    def _post_process_instance(self, outputs, target_size) -> List[Segment]:
        """Post-process instance segmentation."""
        results = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=target_size
        )
        res = results[0]
        
        segments = []
        
        if "segmentation" in res and "segments_info" in res:
            segmentation = res["segmentation"]
            segments_info = res["segments_info"]
            
            for info in segments_info:
                seg_id = info.get("id", 0)
                mask = (segmentation == seg_id).astype(np.uint8)
                
                if mask.sum() == 0:
                    continue
                
                bbox = BoundingBox.from_mask(mask)
                score = float(info.get("score", 1.0))
                label_id = info.get("label_id")
                label = self._get_label_name(label_id) if label_id is not None else "unknown"
                
                segments.append(Segment(
                    mask=mask,
                    bbox=bbox,
                    score=score,
                    label=label,
                    label_id=label_id
                ))
        
        return segments
    
    def _post_process_semantic(self, outputs, target_size) -> List[Segment]:
        """Post-process semantic segmentation."""
        results = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=target_size
        )
        segmap = np.asarray(results[0]).astype(np.int32)
        
        segments = []
        for label_id in np.unique(segmap):
            if label_id < 0:
                continue
            
            mask = (segmap == label_id).astype(np.uint8)
            if mask.sum() == 0:
                continue
            
            bbox = BoundingBox.from_mask(mask)
            label = self._get_label_name(int(label_id))
            
            segments.append(Segment(
                mask=mask,
                bbox=bbox,
                score=1.0,
                label=label,
                label_id=int(label_id)
            ))
        
        return segments
    
    def _post_process_panoptic(self, outputs, target_size) -> List[Segment]:
        """Post-process panoptic segmentation."""
        results = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=target_size
        )
        res = results[0]
        
        segmentation = res["segmentation"]
        segments_info = res["segments_info"]
        
        segments = []
        for info in segments_info:
            seg_id = info.get("id")
            if seg_id is None:
                continue
            
            mask = (segmentation == seg_id).astype(np.uint8)
            if mask.sum() == 0:
                continue
            
            bbox = BoundingBox.from_mask(mask)
            label_id = info.get("label_id")
            label = self._get_label_name(label_id) if label_id is not None else "unknown"
            score = float(info.get("score", 1.0))
            
            segments.append(Segment(
                mask=mask,
                bbox=bbox,
                score=score,
                label=label,
                label_id=label_id,
                metadata={"is_thing": info.get("isthing", False)}
            ))
        
        return segments
    
    def _get_label_name(self, label_id: int) -> str:
        """Get label name from label ID."""
        cfg = getattr(self.model, "config", None)
        if cfg and hasattr(cfg, "id2label"):
            return cfg.id2label.get(label_id, str(label_id))
        return str(label_id)