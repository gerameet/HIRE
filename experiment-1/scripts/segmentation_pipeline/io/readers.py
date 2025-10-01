import os
from typing import List, Tuple
from PIL import Image


class ImageReader:
    """Utility class for reading images."""

    @staticmethod
    def read(path: str) -> Image.Image:
        """Read image from path."""
        with Image.open(path) as img:
            return img.convert("RGB")

    @staticmethod
    def read_batch(paths: List[str]) -> List[Image.Image]:
        """Read multiple images."""
        return [ImageReader.read(p) for p in paths]


def read_image(path: str) -> Image.Image:
    """Read a single image."""
    return ImageReader.read(path)


def read_image_list(list_file: str, root_dir: str = "") -> List[Tuple[str, str]]:
    """Read list of image paths from text file.

    Args:
        list_file: Path to text file with one image path per line
        root_dir: Root directory to prepend to paths

    Returns:
        List of (full_path, relative_path) tuples
    """
    with open(list_file, "r", encoding="utf-8") as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    result = []
    for rel_path in rel_paths:
        full_path = os.path.join(root_dir, rel_path) if root_dir else rel_path
        result.append((full_path, rel_path))

    return result
