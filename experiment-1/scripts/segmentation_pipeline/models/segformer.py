from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from ..core.base import SegmentationModel, ModelConfig
from ..core.data import Segment, BoundingBox
from ..core.utils import ensure_image_pil, get_device
from . import register_model


@register_model("segformer")
class SegFormerSegmentationModel(SegmentationModel):
    """Wrapper for Hugging Face SegFormer models.
    
    Semantic segmentation only.
    
    Requires: pip install transformers torch
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.processor = None
        self.model = None
    
    def initialize(self) -> None:
        """Initialize SegFormer model."""
        try:
            import torch
            from transformers import (
                SegformerForSemanticSegmentation,
                AutoImageProcessor
            )
        except ImportError as e:
            raise RuntimeError(
                "SegFormer requires 'transformers' and 'torch'. Install with:\n"
                "pip install transformers torch"
            ) from e
        
        checkpoint = self.config.checkpoint or "nvidia/segformer-b0-finetuned-ade-512-512"
        device = get_device(self.config.device)
        
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint)
        self.model.to(device)
        self.model.eval()
        
        self._initialized = True
        self.torch = torch
    
    def segment(
        self,
        image: Union[Image.Image, np.ndarray],
        prompts: Optional[Dict[str, Any]] = None
    ) -> List[Segment]:
        """Run SegFormer segmentation."""
        image = ensure_image_pil(image).convert("RGB")
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(get_device(self.config.device)) for k, v in inputs.items()}
        
        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_size = [image.size[::-1]]
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
    
    def _get_label_name(self, label_id: int) -> str:
        """Get label name from label ID."""
        cfg = getattr(self.model, "config", None)
        if cfg and hasattr(cfg, "id2label"):
            return cfg.id2label.get(label_id, str(label_id))
        return str(label_id)