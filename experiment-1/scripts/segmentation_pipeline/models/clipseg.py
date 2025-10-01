from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from ..core.base import TextPromptableSegmentationModel, ModelConfig
from ..core.data import Segment, BoundingBox
from ..core.utils import ensure_image_pil, get_device
from . import register_model


@register_model("clipseg")
class CLIPSegModel(TextPromptableSegmentationModel):
    """Wrapper for CLIPSeg - text-prompted segmentation.

    Allows segmentation based on text descriptions.

    Requires: pip install transformers torch
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.processor = None
        self.model = None

    def initialize(self) -> None:
        """Initialize CLIPSeg model."""
        try:
            import torch
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        except ImportError as e:
            raise RuntimeError(
                "CLIPSeg requires 'transformers' and 'torch'. Install with:\n"
                "pip install transformers torch"
            ) from e

        checkpoint = self.config.checkpoint or "CIDAS/clipseg-rd64-refined"
        device = get_device(self.config.device)

        self.processor = CLIPSegProcessor.from_pretrained(checkpoint)
        self.model = CLIPSegForImageSegmentation.from_pretrained(checkpoint)
        self.model.to(device)
        self.model.eval()

        self._initialized = True
        self.torch = torch

    def segment(
        self,
        image: Union[Image.Image, np.ndarray],
        prompts: Optional[Dict[str, Any]] = None,
    ) -> List[Segment]:
        """Run CLIPSeg segmentation with text prompts."""
        if prompts is None or "text" not in prompts:
            raise ValueError("CLIPSeg requires text prompts in prompts['text']")

        text_prompts = prompts["text"]
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        return self.segment_with_text(image, text_prompts)

    def segment_with_text(
        self, image: Union[Image.Image, np.ndarray], text_prompts: Union[str, List[str]]
    ) -> List[Segment]:
        """Segment image based on text descriptions."""
        image = ensure_image_pil(image).convert("RGB")

        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        # Prepare inputs
        inputs = self.processor(
            text=text_prompts,
            images=[image] * len(text_prompts),
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(get_device(self.config.device)) for k, v in inputs.items()}

        # Run inference
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        # Get predictions
        logits = outputs.logits  # Shape: (num_prompts, H, W)

        segments = []
        threshold = self.config.extra_params.get("threshold", 0.5)

        for i, text_prompt in enumerate(text_prompts):
            # Get mask for this prompt
            pred_mask = self.torch.sigmoid(logits[i])
            pred_mask = pred_mask.cpu().numpy()

            # Resize to original image size
            mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
            mask_img = mask_img.resize(image.size, Image.BILINEAR)
            mask_array = (np.array(mask_img) / 255.0 > threshold).astype(np.uint8)

            if mask_array.sum() == 0:
                continue

            bbox = BoundingBox.from_mask(mask_array)

            # Use mean activation as confidence score
            score = float(pred_mask.mean())

            segments.append(
                Segment(
                    mask=mask_array,
                    bbox=bbox,
                    score=score,
                    label=text_prompt,
                    metadata={"text_prompt": text_prompt},
                )
            )

        return segments
