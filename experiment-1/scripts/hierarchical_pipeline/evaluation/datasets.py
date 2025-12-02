"""Dataset utilities for loading and processing image datasets.

Supports COCO, ImageNet subsets, and custom image directories.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterator, Optional
import logging

logger = logging.getLogger(__name__)


class COCOSubset:
    """Load COCO validation subset for benchmarking.

    Provides iterator over images and annotations.
    """

    def __init__(
        self,
        coco_dir: str,
        num_images: int = 50,
        categories: Optional[List[str]] = None,
        split: str = "val2017",
    ):
        """Initialize COCO subset loader.

        Args:
            coco_dir: Path to COCO dataset directory
            num_images: Number of images to load
            categories: Filter by categories (None = all)
            split: Dataset split (val2017, train2017, etc.)
        """
        self.coco_dir = Path(coco_dir)
        self.num_images = num_images
        self.categories = categories
        self.split = split

        self.images = []
        self.annotations = {}

        # Try to load COCO annotations
        self._load_data()

    def _load_data(self):
        """Load COCO annotations and filter images."""
        ann_file = self.coco_dir / "annotations" / f"instances_{self.split}.json"

        if not ann_file.exists():
            logger.warning(f"COCO annotations not found: {ann_file}")
            logger.info("Falling back to image discovery mode")
            self._discover_images()
            return

        try:
            with open(ann_file) as f:
                coco_data = json.load(f)

            # Build category mapping if filtering
            cat_ids = None
            if self.categories:
                cat_ids = {
                    cat["id"]
                    for cat in coco_data["categories"]
                    if cat["name"] in self.categories
                }

            # Filter images
            image_ids = set()
            for ann in coco_data["annotations"]:
                if cat_ids is None or ann["category_id"] in cat_ids:
                    image_ids.add(ann["image_id"])

                    if len(image_ids) >= self.num_images:
                        break

            # Get image paths
            for img in coco_data["images"]:
                if img["id"] in image_ids:
                    img_path = self.coco_dir / self.split / img["file_name"]
                    if img_path.exists():
                        self.images.append(str(img_path))

                    if len(self.images) >= self.num_images:
                        break

            logger.info(f"Loaded {len(self.images)} images from COCO {self.split}")

        except Exception as e:
            logger.error(f"Failed to load COCO annotations: {e}")
            self._discover_images()

    def _discover_images(self):
        """Discover images in directory (fallback when annotations unavailable)."""
        img_dir = self.coco_dir / self.split
        if not img_dir.exists():
            img_dir = self.coco_dir

        exts = {".jpg", ".jpeg", ".png"}
        self.images = [
            str(p) for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts
        ][: self.num_images]

        logger.info(f"Discovered {len(self.images)} images in {img_dir}")

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Iterate over (image_path, annotations) pairs."""
        for img_path in self.images:
            # Return image path and empty annotations for now
            yield img_path, {}

    def __len__(self) -> int:
        return len(self.images)


class CustomDataset:
    """Load images from a custom directory.

    Auto-discovers all images in the directory.
    """

    def __init__(self, image_dir: str, num_images: Optional[int] = None):
        """Initialize custom dataset.

        Args:
            image_dir: Directory containing images
            num_images: Maximum number of images (None = all)
        """
        self.image_dir = Path(image_dir)
        self.num_images = num_images

        self.images = self._discover_images()

    def _discover_images(self) -> List[str]:
        """Discover all images in directory."""
        if not self.image_dir.exists():
            logger.error(f"Directory not found: {self.image_dir}")
            return []

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = [
            str(p)
            for p in sorted(self.image_dir.rglob("*"))
            if p.suffix.lower() in exts and p.is_file()
        ]

        if self.num_images:
            images = images[: self.num_images]

        logger.info(f"Found {len(images)} images in {self.image_dir}")
        return images

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Iterate over (image_path, annotations) pairs."""
        for img_path in self.images:
            yield img_path, {}

    def __len__(self) -> int:
        return len(self.images)


class SyntheticDataset:
    """Generate synthetic images with ground truth hierarchies.

    Useful for testing and prototyping without real data.
    """

    def __init__(self, num_images: int = 10, image_size: Tuple[int, int] = (224, 224)):
        """Initialize synthetic dataset generator.

        Args:
            num_images: Number of synthetic images to generate
            image_size: Size of generated images (H, W)
        """
        self.num_images = num_images
        self.image_size = image_size

    def generate_image(self, seed: int) -> Tuple[Any, List[Any]]:
        """Generate a synthetic image with parts.

        Args:
            seed: Random seed for reproducibility

        Returns:
            (image, parts) tuple
        """
        import numpy as np

        # Set seed
        np.random.seed(seed)

        # Generate random image with geometric shapes
        h, w = self.image_size
        image = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)

        # Generate 3-5 random parts
        num_parts = np.random.randint(3, 6)
        parts = []

        for i in range(num_parts):
            # Random position and size
            size = np.random.randint(20, 60)
            x = np.random.randint(0, w - size)
            y = np.random.randint(0, h - size)

            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y : y + size, x : x + size] = 1

            # Create simple Part-like object
            part = {
                "id": f"synthetic_{seed}_{i}",
                "mask": mask,
                "bbox": (x, y, x + size, y + size),
                "area": size * size,
            }
            parts.append(part)

        return image, parts

    def __iter__(self) -> Iterator[Tuple[Any, List[Any]]]:
        """Iterate over synthetic (image, parts) pairs."""
        for i in range(self.num_images):
            yield self.generate_image(i)

    def __len__(self) -> int:
        return self.num_images


def load_dataset(dataset_type: str, **kwargs) -> Any:
    """Factory function to load datasets.

    Args:
        dataset_type: Type of dataset ("coco", "custom", "synthetic")
        **kwargs: Arguments for dataset constructor

    Returns:
        Dataset object
    """
    if dataset_type.lower() == "coco":
        return COCOSubset(**kwargs)
    elif dataset_type.lower() == "custom":
        return CustomDataset(**kwargs)
    elif dataset_type.lower() == "synthetic":
        return SyntheticDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
