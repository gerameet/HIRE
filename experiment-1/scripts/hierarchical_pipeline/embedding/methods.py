"""Embedding generation methods for visual parts.

Implements multiple embedding strategies using pre-trained models:
- DummyEmbedding: Random embeddings for testing/baseline
- DINOEmbedding: Self-supervised ViT (facebook/dino-vitb16)
- CLIPEmbedding: Vision-language model (openai/clip-vit-base-patch32)
- MAEEmbedding: Masked autoencoder (facebook/mae-vit-base)
"""

import numpy as np
from typing import List, Optional, Dict, Any
from PIL import Image
import logging

from ..core.interfaces import EmbeddingMethod
from .cache import EmbeddingCache

logger = logging.getLogger(__name__)

# Try to import torch and transformers
try:
    import torch
    import torchvision.transforms as T
    from transformers import AutoModel, AutoProcessor, AutoImageProcessor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch or transformers not available. Only DummyEmbedding will work.")


class DummyEmbedding(EmbeddingMethod):
    """Random embedding for testing and baseline comparisons.
    
    Generates deterministic random embeddings using a seed for reproducibility.
    Useful for testing pipeline without waiting for model inference.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize dummy embedding generator.
        
        Args:
            config: Configuration with optional 'seed' and 'embedding_dim'
        """
        super().__init__(config)
        self.seed = config.get("seed", 42)
        self.embedding_dim = config.get("embedding_dim", 768)
        self.rng = np.random.RandomState(self.seed)
        
        logger.info(f"DummyEmbedding initialized (dim={self.embedding_dim}, seed={self.seed})")
    
    def embed_part(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate random embedding.
        
        Args:
            image: Full image (H, W, C)
            mask: Binary mask (H, W)
            
        Returns:
            Random normalized embedding vector
        """
        # Generate random embedding
        embedding = self.rng.randn(self.embedding_dim)
        
        # Normalize to unit length
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding.astype(np.float32)
    
    def get_embedding_space(self) -> str:
        """Return embedding space type."""
        return "euclidean"


class DINOEmbedding(EmbeddingMethod):
    """DINO self-supervised ViT embedding.
    
    Uses facebook/dino-vitb16 or similar DINO models for robust visual features.
    DINO is trained with self-distillation and produces semantically meaningful
    embeddings without labels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DINO embedding generator.
        
        Args:
            config: Configuration with 'model_name', 'device', etc.
        """
        super().__init__(config)
        
        if not HAS_TORCH:
            raise ImportError("PyTorch and transformers required for DINOEmbedding")
        
        self.model_name = config.get("model_name", "facebook/dino-vitb16")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.get("batch_size", 32)
        self.use_cache = config.get("use_cache", True)
        self.cache_dir = config.get("cache_dir", "cache/embeddings")
        
        # Initialize cache
        self.cache = EmbeddingCache(self.cache_dir, enabled=self.use_cache)
        
        # Load model
        logger.info(f"Loading DINO model: {self.model_name}")
        try:
            # Try Facebook DINO models first
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=True)
            self.model.eval()
            self.model.to(self.device)
            
            # DINO preprocessing
            self.transform = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            
            # Get embedding dimension
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                dummy_output = self.model(dummy_input)
                self.embedding_dim = dummy_output.shape[-1]
                
            logger.info(f"DINO model loaded (dim={self.embedding_dim}, device={self.device})")
            
        except Exception as e:
            logger.error(f"Failed to load DINO model: {e}")
            raise
    
    def _extract_part_image(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """Extract masked region from image.
        
        Args:
            image: Full image (H, W, C) as numpy array
            mask: Binary mask (H, W)
            
        Returns:
            PIL Image of cropped region
        """
        # Find bounding box of mask
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            # Empty mask - return small blank image
            return Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        y1, y2 = rows.min(), rows.max() + 1
        x1, x2 = cols.min(), cols.max() + 1
        
        # Crop image to bbox
        cropped = image[y1:y2, x1:x2]
        
        # Convert to PIL
        if cropped.dtype == np.float32 or cropped.dtype == np.float64:
            cropped = (cropped * 255).astype(np.uint8)
        
        return Image.fromarray(cropped)
    
    def embed_part(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate DINO embedding for masked region.
        
        Args:
            image: Full image (H, W, C)
            mask: Binary mask (H, W)
            
        Returns:
            DINO embedding vector (normalized)
        """
        # Extract part image
        part_img = self._extract_part_image(image, mask)
        
        # Preprocess
        img_tensor = self.transform(part_img).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            embedding = self.model(img_tensor)
        
        # Convert to numpy
        embedding = embedding.cpu().numpy()[0]
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding.astype(np.float32)
    
    def embed_batch(
        self, 
        images: List[np.ndarray], 
        masks: List[np.ndarray]
    ) -> np.ndarray:
        """Batch embedding generation (optimized for GPU).
        
        Args:
            images: List of full images
            masks: List of binary masks
            
        Returns:
            Embedding matrix (N, embedding_dim)
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_masks = masks[i:i + self.batch_size]
            
            # Extract part images
            part_images = [
                self._extract_part_image(img, mask)
                for img, mask in zip(batch_images, batch_masks)
            ]
            
            # Preprocess batch
            batch_tensors = torch.stack([
                self.transform(img) for img in part_images
            ]).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensors)
            
            # Convert to numpy
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            # Normalize
            batch_norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            batch_embeddings = batch_embeddings / (batch_norms + 1e-8)
            
            embeddings.append(batch_embeddings)
        
        return np.concatenate(embeddings, axis=0).astype(np.float32)
    
    def get_embedding_space(self) -> str:
        """Return embedding space type."""
        return "euclidean"


class CLIPEmbedding(EmbeddingMethod):
    """CLIP vision-language embedding.
    
    Uses OpenAI CLIP for embeddings aligned with text space.
    Can be used for zero-shot concept alignment later.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CLIP embedding generator.
        
        Args:
            config: Configuration with 'model_name', 'device', etc.
        """
        super().__init__(config)
        
        if not HAS_TORCH:
            raise ImportError("PyTorch and transformers required for CLIPEmbedding")
        
        self.model_name = config.get("model_name", "openai/clip-vit-base-patch32")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.get("batch_size", 32)
        self.use_cache = config.get("use_cache", True)
        self.cache_dir = config.get("cache_dir", "cache/embeddings")
        
        self.cache = EmbeddingCache(self.cache_dir, enabled=self.use_cache)
        
        logger.info(f"Loading CLIP model: {self.model_name}")
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            self.model.eval()
            self.model.to(self.device)
            
            # Get embedding dimension from vision model
            self.embedding_dim = self.model.config.projection_dim
            
            logger.info(f"CLIP model loaded (dim={self.embedding_dim}, device={self.device})")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _extract_part_image(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """Extract masked region (same as DINO)."""
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        y1, y2 = rows.min(), rows.max() + 1
        x1, x2 = cols.min(), cols.max() + 1
        cropped = image[y1:y2, x1:x2]
        
        if cropped.dtype == np.float32 or cropped.dtype == np.float64:
            cropped = (cropped * 255).astype(np.uint8)
        
        return Image.fromarray(cropped)
    
    def embed_part(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate CLIP embedding for masked region.
        
        Args:
            image: Full image (H, W, C)
            mask: Binary mask (H, W)
            
        Returns:
            CLIP vision embedding (normalized)
        """
        part_img = self._extract_part_image(image, mask)
        
        # Preprocess with CLIP processor
        inputs = self.processor(images=part_img, return_tensors="pt").to(self.device)
        
        # Forward pass (vision encoder only)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        
        # Convert to numpy
        embedding = embedding.cpu().numpy()[0]
        
        # Normalize (CLIP embeddings are already normalized, but ensure it)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding.astype(np.float32)
    
    def get_embedding_space(self) -> str:
        """Return embedding space type."""
        return "euclidean"


class MAEEmbedding(EmbeddingMethod):
    """Masked Autoencoder (MAE) embedding.
    
    Uses facebook/mae-vit-base encoder for fine-grained features.
    MAE is trained to reconstruct masked patches, learning good representations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MAE embedding generator.
        
        Args:
            config: Configuration with 'model_name', 'device', etc.
        """
        super().__init__(config)
        
        if not HAS_TORCH:
            raise ImportError("PyTorch and transformers required for MAEEmbedding")
        
        self.model_name = config.get("model_name", "facebook/vit-mae-base")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.get("batch_size", 32)
        self.use_cache = config.get("use_cache", True)
        self.cache_dir = config.get("cache_dir", "cache/embeddings")
        
        self.cache = EmbeddingCache(self.cache_dir, enabled=self.use_cache)
        
        logger.info(f"Loading MAE model: {self.model_name}")
        try:
            from transformers import ViTMAEModel, ViTImageProcessor
            
            self.model = ViTMAEModel.from_pretrained(self.model_name)
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            
            self.model.eval()
            self.model.to(self.device)
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            
            logger.info(f"MAE model loaded (dim={self.embedding_dim}, device={self.device})")
            
        except Exception as e:
            logger.error(f"Failed to load MAE model: {e}")
            raise
    
    def _extract_part_image(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """Extract masked region (same as DINO)."""
        rows, cols = np.where(mask > 0)
        if len(rows) == 0:
            return Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        y1, y2 = rows.min(), rows.max() + 1
        x1, x2 = cols.min(), cols.max() + 1
        cropped = image[y1:y2, x1:x2]
        
        if cropped.dtype == np.float32 or cropped.dtype == np.float64:
            cropped = (cropped * 255).astype(np.uint8)
        
        return Image.fromarray(cropped)
    
    def embed_part(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Generate MAE embedding for masked region.
        
        Args:
            image: Full image (H, W, C)
            mask: Binary mask (H, W)
            
        Returns:
            MAE embedding (CLS token, normalized)
        """
        part_img = self._extract_part_image(image, mask)
        
        # Preprocess
        inputs = self.processor(images=part_img, return_tensors="pt").to(self.device)
        
        # Forward pass (encoder only, take CLS token)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token (first token)
            embedding = outputs.last_hidden_state[:, 0, :]
        
        # Convert to numpy
        embedding = embedding.cpu().numpy()[0]
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding.astype(np.float32)
    
    def get_embedding_space(self) -> str:
        """Return embedding space type."""
        return "euclidean"
