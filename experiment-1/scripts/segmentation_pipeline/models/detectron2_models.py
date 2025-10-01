from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from ..core.base import SegmentationModel, ModelConfig
from ..core.data import Segment, BoundingBox
from ..core.utils import ensure_image_pil, normalize_image
from . import register_model


@register_model("detectron2")
class Detectron2SegmentationModel(SegmentationModel):
    """Wrapper for Detectron2 models.
    
    Supports various architectures:
    - Mask R-CNN
    - Cascade Mask R-CNN
    - PointRend
    - etc.
    
    Requires: pip install detectron2
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.predictor = None
        self.metadata = None
    
    def initialize(self) -> None:
        """Initialize Detectron2 model."""
        try:
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            from detectron2.data import MetadataCatalog
        except ImportError as e:
            raise RuntimeError(
                "Detectron2 required. Install with:\n"
                "python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            ) from e
        
        cfg = get_cfg()
        
        # Load config from model zoo or custom config
        config_file = self.config.extra_params.get("config_file")
        if config_file:
            cfg.merge_from_file(config_file)
        else:
            # Default: Mask R-CNN R50 FPN
            cfg.merge_from_file(
                model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            )
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        # Override with custom checkpoint if provided
        if self.config.checkpoint:
            cfg.MODEL.WEIGHTS = self.config.checkpoint
        
        # Set device
        cfg.MODEL.DEVICE = self.config.device or "cuda"
        
        # Set confidence threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.confidence_threshold
        
        self.predictor = DefaultPredictor(cfg)
        
        # Get metadata for class names
        dataset_name = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else "coco_2017_train"
        self.metadata = MetadataCatalog.get(dataset_name)
        
        self._initialized = True
    
    def segment(
        self,
        image: Union[Image.Image, np.ndarray],
        prompts: Optional[Dict[str, Any]] = None
    ) -> List[Segment]:
        """Run Detectron2 segmentation."""
        if isinstance(image, Image.Image):
            np_img = normalize_image(image)
        else:
            np_img = image
        
        # Run prediction
        outputs = self.predictor(np_img)
        
        instances = outputs["instances"].to("cpu")
        
        segments = []
        
        if not instances.has("pred_masks"):
            return segments
        
        pred_masks = instances.pred_masks.numpy()
        pred_boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()
        
        # Get class names
        class_names = self.metadata.get("thing_classes", [])
        
        for mask, box, score, cls_id in zip(pred_masks, pred_boxes, scores, pred_classes):
            mask_uint8 = mask.astype(np.uint8)
            
            x1, y1, x2, y2 = box
            bbox = BoundingBox(
                int(x1), int(y1), int(x2), int(y2)
            )
            
            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            
            segments.append(Segment(
                mask=mask_uint8,
                bbox=bbox,
                score=float(score),
                label=label,
                label_id=int(cls_id)
            ))
        
        return segments