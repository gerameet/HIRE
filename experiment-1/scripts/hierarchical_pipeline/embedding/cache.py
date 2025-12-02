"""Embedding cache system for fast iteration.

Caches embeddings by (image_path, part_id, embedding_method) to avoid
recomputation when experimenting with different hierarchy builders.
"""

import os
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Disk-based cache for part embeddings.

    Cache structure:
        cache_dir/
            <hash>/
                metadata.json  # {image_path, part_id, method, config}
                embedding.npy  # numpy array
    """

    def __init__(self, cache_dir: str = "cache/embeddings", enabled: bool = True):
        """Initialize embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings
            enabled: If False, cache is disabled (always miss)
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache initialized at {self.cache_dir}")

    def _compute_key(
        self,
        image_path: str,
        part_id: str,
        method_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Compute cache key hash.

        Args:
            image_path: Path to image
            part_id: Unique part identifier
            method_name: Embedding method name
            config: Method configuration (for cache invalidation)

        Returns:
            Hexadecimal hash string
        """
        # Create deterministic config string
        config_str = json.dumps(config or {}, sort_keys=True)

        # Combine all components
        key_string = f"{image_path}|{part_id}|{method_name}|{config_str}"

        # Hash to fixed length
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def get(
        self,
        image_path: str,
        part_id: str,
        method_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """Retrieve cached embedding.

        Args:
            image_path: Path to image
            part_id: Unique part identifier
            method_name: Embedding method name
            config: Method configuration

        Returns:
            Cached embedding array or None if not found
        """
        if not self.enabled:
            return None

        cache_key = self._compute_key(image_path, part_id, method_name, config)
        cache_path = self.cache_dir / cache_key

        if not cache_path.exists():
            return None

        try:
            # Load embedding
            embedding = np.load(cache_path / "embedding.npy")
            logger.debug(f"Cache hit: {cache_key}")
            return embedding

        except Exception as e:
            logger.warning(f"Failed to load cached embedding {cache_key}: {e}")
            return None

    def put(
        self,
        image_path: str,
        part_id: str,
        method_name: str,
        embedding: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store embedding in cache.

        Args:
            image_path: Path to image
            part_id: Unique part identifier
            method_name: Embedding method name
            embedding: Embedding array to cache
            config: Method configuration
        """
        if not self.enabled:
            return

        cache_key = self._compute_key(image_path, part_id, method_name, config)
        cache_path = self.cache_dir / cache_key
        cache_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save embedding
            np.save(cache_path / "embedding.npy", embedding)

            # Save metadata for debugging
            metadata = {
                "image_path": image_path,
                "part_id": part_id,
                "method_name": method_name,
                "config": config or {},
                "embedding_shape": list(embedding.shape),
                "embedding_dtype": str(embedding.dtype),
            }

            with open(cache_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Cache put: {cache_key}")

        except Exception as e:
            logger.warning(f"Failed to cache embedding {cache_key}: {e}")

    def clear(self) -> None:
        """Clear all cached embeddings."""
        if not self.enabled or not self.cache_dir.exists():
            return

        import shutil

        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Embedding cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache size, count, etc.
        """
        if not self.enabled or not self.cache_dir.exists():
            return {"enabled": False}

        # Count cached entries
        entries = list(self.cache_dir.iterdir())
        total_size = sum(
            sum(f.stat().st_size for f in entry.iterdir() if f.is_file())
            for entry in entries
            if entry.is_dir()
        )

        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "num_entries": len(entries),
            "total_size_mb": total_size / (1024 * 1024),
        }
