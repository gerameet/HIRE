"""Centralized model downloading and management system.

Handles downloading, caching, and verification of all models used in the pipeline:
- HuggingFace transformers models (CLIP, MAE, Mask2Former, etc.)
- Torch hub models (DINO, DINOv2)
- Manual downloads (SAM checkpoints)
- Ultralytics models (YOLO)
"""

import json
import hashlib
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Specification for a model."""
    
    name: str
    type: str  # 'huggingface', 'torch_hub', 'manual', 'ultralytics'
    source: str  # URL, HF model ID, or torch hub repo
    required_for: List[str]  # Which methods need this model
    size_mb: float
    local_path: Optional[str] = None
    checksum: Optional[str] = None
    model_fn: Optional[str] = None  # For torch hub models
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelSpec':
        """Create from dictionary."""
        return cls(**data)


class ModelManager:
    """Manages downloading, caching, and verification of all models.
    
    Usage:
        manager = ModelManager()
        
        # Check if model is available
        if not manager.is_available("sam-vit-b"):
            manager.download_model("sam-vit-b")
        
        # Get path to model
        path = manager.get_model_path("sam-vit-b")
        
        # List all models
        models = manager.list_available_models()
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize model manager.
        
        Args:
            cache_dir: Directory for model cache. Defaults to cache/models/
        """
        if cache_dir is None:
            # Default to cache/models relative to script root
            script_root = Path(__file__).parent.parent.parent.parent
            cache_dir = script_root / "cache" / "models"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model registry
        self.registry = self._load_registry()
        
        logger.info(f"ModelManager initialized (cache: {self.cache_dir})")
    
    def _load_registry(self) -> Dict[str, ModelSpec]:
        """Load model registry from JSON file."""
        registry_path = Path(__file__).parent / "registry.json"
        
        if not registry_path.exists():
            logger.warning(f"Model registry not found at {registry_path}")
            return {}
        
        try:
            with open(registry_path) as f:
                data = json.load(f)
            
            registry = {}
            for name, spec_dict in data.get("models", {}).items():
                registry[name] = ModelSpec.from_dict({**spec_dict, "name": name})
            
            logger.info(f"Loaded {len(registry)} model specifications")
            return registry
            
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            return {}
    
    def list_available_models(self) -> Dict[str, bool]:
        """List all models and their availability.
        
        Returns:
            Dictionary mapping model name to availability (True/False)
        """
        return {
            name: self.is_available(name)
            for name in self.registry.keys()
        }
    
    def is_available(self, model_name: str) -> bool:
        """Check if model is available locally.
        
        Args:
            model_name: Name of model (e.g., "sam-vit-b")
            
        Returns:
            True if model is cached and verified
        """
        if model_name not in self.registry:
            logger.warning(f"Unknown model: {model_name}")
            return False
        
        spec = self.registry[model_name]
        
        # For HuggingFace and torch_hub, check if files exist
        if spec.type in ["huggingface", "torch_hub"]:
            model_cache = self._get_cache_path(model_name)
            return model_cache.exists() and any(model_cache.iterdir())
        
        # For manual downloads, check specific file
        elif spec.type == "manual":
            if spec.local_path:
                path = self.cache_dir / spec.local_path
                return path.exists()
        
        # For ultralytics, check default location
        elif spec.type == "ultralytics":
            # Ultralytics downloads to ~/.cache/ultralytics or specified location
            return True  # Assume available, will auto-download
        
        return False
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get local path to model.
        
        Args:
            model_name: Name of model
            
        Returns:
            Path to model or None if not available
        """
        if model_name not in self.registry:
            raise ValueError(f"Unknown model: {model_name}")
        
        spec = self.registry[model_name]
        
        if spec.type == "manual" and spec.local_path:
            path = self.cache_dir / spec.local_path
            if path.exists():
                return path
        elif spec.type in ["huggingface", "torch_hub"]:
            return self._get_cache_path(model_name)
        
        return None
    
    def _get_cache_path(self, model_name: str) -> Path:
        """Get cache directory for a model."""
        spec = self.registry[model_name]
        if spec.type == "huggingface":
            # Use subdirectory for HF models
            return self.cache_dir / "huggingface" / spec.source.replace("/", "--")
        elif spec.type == "torch_hub":
            return self.cache_dir / "torch_hub" / model_name
        return self.cache_dir / model_name
    
    def download_model(self, model_name: str, force: bool = False) -> Path:
        """Download a model if not cached.
        
        Args:
            model_name: Name of model to download
            force: Force re-download even if cached
            
        Returns:
            Path to downloaded model
        """
        if model_name not in self.registry:
            raise ValueError(f"Unknown model: {model_name}")
        
        spec = self.registry[model_name]
        
        # Check if already available
        if not force and self.is_available(model_name):
            logger.info(f"Model {model_name} already available")
            return self.get_model_path(model_name)
        
        logger.info(f"Downloading {model_name} ({spec.size_mb:.1f} MB)...")
        
        if spec.type == "manual":
            return self._download_manual(spec)
        elif spec.type == "huggingface":
            return self._download_huggingface(spec)
        elif spec.type == "torch_hub":
            return self._download_torch_hub(spec)
        elif spec.type == "ultralytics":
            return self._download_ultralytics(spec)
        else:
            raise ValueError(f"Unknown model type: {spec.type}")
    
    def _download_manual(self, spec: ModelSpec) -> Path:
        """Download model from direct URL."""
        if not spec.local_path:
            raise ValueError(f"No local_path specified for {spec.name}")
        
        output_path = self.cache_dir / spec.local_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        response = requests.get(spec.source, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=spec.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Verify checksum if provided
        if spec.checksum:
            if not self._verify_checksum(output_path, spec.checksum):
                output_path.unlink()
                raise RuntimeError(f"Checksum verification failed for {spec.name}")
        
        logger.info(f"Downloaded {spec.name} to {output_path}")
        return output_path
    
    def _download_huggingface(self, spec: ModelSpec) -> Path:
        """Download HuggingFace model."""
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise ImportError("transformers required for HuggingFace models")
        
        cache_path = self._get_cache_path(spec.name)
        
        # Download model and processor
        try:
            logger.info(f"Downloading HuggingFace model: {spec.source}")
            AutoModel.from_pretrained(spec.source, cache_dir=str(cache_path))
            AutoProcessor.from_pretrained(spec.source, cache_dir=str(cache_path))
        except Exception as e:
            logger.error(f"Failed to download {spec.name}: {e}")
            raise
        
        return cache_path
    
    def _download_torch_hub(self, spec: ModelSpec) -> Path:
        """Download torch hub model."""
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for torch hub models")
        
        cache_path = self._get_cache_path(spec.name)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Set torch hub cache directory
        torch.hub.set_dir(str(cache_path))
        
        # Load model (will download if needed)
        if spec.model_fn:
            logger.info(f"Loading torch hub model: {spec.source}/{spec.model_fn}")
            torch.hub.load(spec.source, spec.model_fn, pretrained=True)
        else:
            raise ValueError(f"No model_fn specified for torch hub model {spec.name}")
        
        return cache_path
    
    def _download_ultralytics(self, spec: ModelSpec) -> Path:
        """Download Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics required for YOLO models")
        
        # Ultralytics handles downloading automatically
        logger.info(f"Loading YOLO model: {spec.source}")
        model = YOLO(spec.source)
        
        # Get actual model path
        return Path(model.ckpt_path) if hasattr(model, 'ckpt_path') else self.cache_dir
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum (MD5).
        
        Args:
            file_path: Path to file
            expected_checksum: Expected MD5 checksum
            
        Returns:
            True if checksum matches
        """
        md5_hash = hashlib.md5()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        
        actual_checksum = md5_hash.hexdigest()
        return actual_checksum == expected_checksum
    
    def verify_model(self, model_name: str) -> bool:
        """Verify model integrity using checksum.
        
        Args:
            model_name: Name of model
            
        Returns:
            True if model is valid
        """
        if not self.is_available(model_name):
            return False
        
        spec = self.registry[model_name]
        
        # Only verify manual downloads with checksums
        if spec.type == "manual" and spec.checksum:
            model_path = self.get_model_path(model_name)
            return self._verify_checksum(model_path, spec.checksum)
        
        # For other types, just check existence
        return True
    
    def download_all_for_config(self, config: Dict[str, Any]) -> None:
        """Download all models needed for a configuration.
        
        Args:
            config: Experiment configuration dictionary
        """
        required_models = set()
        
        # Check segmentation model
        seg_model = config.get("segmentation", {}).get("model", "")
        for model_name, spec in self.registry.items():
            if seg_model.lower() in spec.required_for:
                required_models.add(model_name)
        
        # Check embedding method
        emb_method = config.get("embedding", {}).get("method", "")
        for model_name, spec in self.registry.items():
            if emb_method.lower() in spec.required_for:
                required_models.add(model_name)
        
        # Download all required models
        for model_name in required_models:
            if not self.is_available(model_name):
                logger.info(f"Downloading required model: {model_name}")
                self.download_model(model_name)
    
    def get_cache_size(self) -> float:
        """Get total size of model cache in MB.
        
        Returns:
            Total size in megabytes
        """
        total_size = sum(
            f.stat().st_size 
            for f in self.cache_dir.rglob('*') 
            if f.is_file()
        )
        return total_size / (1024 * 1024)
    
    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """Clear model cache.
        
        Args:
            model_name: Specific model to clear, or None to clear all
        """
        if model_name:
            if model_name not in self.registry:
                raise ValueError(f"Unknown model: {model_name}")
            
            spec = self.registry[model_name]
            if spec.type == "manual" and spec.local_path:
                path = self.cache_dir / spec.local_path
                if path.exists():
                    path.unlink()
                    logger.info(f"Cleared cache for {model_name}")
            else:
                cache_path = self._get_cache_path(model_name)
                if cache_path.exists():
                    import shutil
                    shutil.rmtree(cache_path)
                    logger.info(f"Cleared cache for {model_name}")
        else:
            # Clear entire cache
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared entire model cache")
