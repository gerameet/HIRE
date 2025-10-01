import os
import time
import sys
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm import tqdm
from PIL import Image

from .core.base import SegmentationModel, ModelConfig
from .core.data import SegmentationResult, Segment
from .core.utils import filter_segments
from .models import get_model, list_available_models
from .io.readers import read_image, read_image_list
from .io.writers import write_segments


@dataclass
class PipelineConfig:
    """Configuration for segmentation pipeline."""

    data_root: str
    output_root: str
    model_name: str
    model_config: ModelConfig

    # Processing options
    num_workers: int = 4
    batch_size: int = 1

    # Filtering options
    min_area: int = 0
    max_area: Optional[int] = None
    min_score: float = 0.0
    filter_labels: Optional[List[str]] = None
    apply_nms: bool = False
    nms_threshold: float = 0.5

    # Output options
    save_arrow: bool = True
    save_crops: bool = False
    save_viz: bool = False
    save_original: bool = False

    # Progress options
    show_progress: bool = True
    verbose: bool = False


class SegmentationPipeline:
    """Main segmentation pipeline with progress tracking."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model: Optional[SegmentationModel] = None
        self.stats = {
            "total_images": 0,
            "successful": 0,
            "failed": 0,
            "total_segments": 0,
            "total_time": 0.0,
            "errors": [],
        }

    def initialize(self) -> None:
        """Initialize the pipeline and model."""
        if self.config.verbose:
            print(f"🚀 Initializing segmentation pipeline...")
            print(f"   Model: {self.config.model_name}")
            print(f"   Available models: {', '.join(list_available_models())}")

        # Create model
        self.model = get_model(self.config.model_name, self.config.model_config)

        # Initialize model
        if self.config.verbose:
            print(f"   Loading model weights...")

        self.model.initialize()

        if self.config.verbose:
            model_info = self.model.get_model_info()
            print(f"   ✓ Model initialized")
            print(f"   Uses GPU: {model_info['uses_gpu']}")
            print(f"   Device: {self.config.model_config.device or 'auto'}")

    def run_on_image_list(self, list_file: str) -> Dict[str, Any]:
        """Run pipeline on list of images from text file.

        Args:
            list_file: Path to text file with image paths (one per line)

        Returns:
            Dictionary with statistics and results
        """
        # Read image list
        image_list = read_image_list(list_file, self.config.data_root)
        self.stats["total_images"] = len(image_list)

        if self.config.verbose:
            print(f"\n📋 Processing {len(image_list)} images")
            print(f"   Data root: {self.config.data_root}")
            print(f"   Output root: {self.config.output_root}")
            print(f"   Workers: {self.config.num_workers}")

        # Adjust workers if GPU model
        if self.model.uses_gpu() and self.config.num_workers > 1:
            print(
                f"⚠️  GPU model detected, reducing workers to 1 to avoid CUDA context issues"
            )
            self.config.num_workers = 1

        # Create output directory
        os.makedirs(self.config.output_root, exist_ok=True)

        # Process images
        start_time = time.time()

        if self.config.num_workers == 1:
            results = self._process_sequential(image_list)
        else:
            results = self._process_parallel(image_list)

        self.stats["total_time"] = time.time() - start_time

        # Print summary
        self._print_summary()

        return {"stats": self.stats, "results": results}

    def run_on_single_image(
        self, image_path: str, output_dir: Optional[str] = None
    ) -> SegmentationResult:
        """Run pipeline on a single image.

        Args:
            image_path: Path to image
            output_dir: Optional output directory (defaults to pipeline output_root)

        Returns:
            SegmentationResult
        """
        if output_dir is None:
            image_name = Path(image_path).stem
            output_dir = os.path.join(self.config.output_root, image_name)

        if self.config.verbose:
            print(f"Processing: {image_path}")

        result = self._process_single_image(image_path, image_path, output_dir)

        if self.config.verbose:
            print(f"✓ Found {len(result.segments)} segments")
            print(f"  Processing time: {result.processing_time:.3f}s")

        return result

    def _process_sequential(self, image_list: List[tuple]) -> List[SegmentationResult]:
        """Process images sequentially with progress bar."""
        results = []

        # Create progress bar
        pbar = tqdm(
            image_list,
            desc="Processing images",
            disable=not self.config.show_progress,
            unit="img",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for full_path, rel_path in pbar:
            try:
                # Get output directory
                image_name = Path(rel_path).stem
                output_dir = os.path.join(self.config.output_root, image_name)

                # Process
                result = self._process_single_image(full_path, rel_path, output_dir)
                results.append(result)

                # Update stats
                self.stats["successful"] += 1
                self.stats["total_segments"] += len(result.segments)

                # Update progress bar
                pbar.set_postfix(
                    {
                        "segments": len(result.segments),
                        "time": f"{result.processing_time:.2f}s",
                        "success_rate": f"{self.stats['successful']}/{self.stats['total_images']}",
                    }
                )

            except Exception as e:
                self.stats["failed"] += 1
                self.stats["errors"].append((rel_path, str(e)))

                if self.config.verbose:
                    print(f"\n❌ Error processing {rel_path}: {e}")

                pbar.set_postfix(
                    {
                        "error": "last failed",
                        "success_rate": f"{self.stats['successful']}/{self.stats['total_images']}",
                    }
                )

        return results

    def _process_parallel(self, image_list: List[tuple]) -> List[SegmentationResult]:
        """Process images in parallel with progress bar."""
        results = []

        # Use ThreadPoolExecutor (good for I/O bound tasks)
        executor = ThreadPoolExecutor(max_workers=self.config.num_workers)

        # Submit all tasks
        future_to_image = {}
        for full_path, rel_path in image_list:
            image_name = Path(rel_path).stem
            output_dir = os.path.join(self.config.output_root, image_name)

            future = executor.submit(
                self._process_single_image, full_path, rel_path, output_dir
            )
            future_to_image[future] = (full_path, rel_path)

        # Process completed tasks with progress bar
        pbar = tqdm(
            total=len(image_list),
            desc="Processing images",
            disable=not self.config.show_progress,
            unit="img",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for future in as_completed(future_to_image):
            full_path, rel_path = future_to_image[future]

            try:
                result = future.result()
                results.append(result)

                # Update stats
                self.stats["successful"] += 1
                self.stats["total_segments"] += len(result.segments)

                # Update progress bar
                pbar.set_postfix(
                    {
                        "segments": len(result.segments),
                        "success_rate": f"{self.stats['successful']}/{self.stats['total_images']}",
                    }
                )

            except Exception as e:
                self.stats["failed"] += 1
                self.stats["errors"].append((rel_path, str(e)))

                if self.config.verbose:
                    print(f"\n❌ Error processing {rel_path}: {e}")

            pbar.update(1)

        pbar.close()
        executor.shutdown()

        return results

    def _process_single_image(
        self, image_path: str, rel_path: str, output_dir: str
    ) -> SegmentationResult:
        """Process a single image."""
        start_time = time.time()

        # Read image
        image = read_image(image_path)

        # Run segmentation
        segments = self.model.segment(image)

        # Apply filtering
        segments = self._filter_segments(segments)

        # Apply NMS if requested
        if self.config.apply_nms:
            from .core.utils import non_max_suppression

            segments = non_max_suppression(segments, self.config.nms_threshold)

        processing_time = time.time() - start_time

        # Create result
        result = SegmentationResult(
            image_path=rel_path,
            segments=segments,
            image_size=image.size,
            processing_time=processing_time,
            model_info=self.model.get_model_info(),
        )

        # Write outputs
        write_segments(
            result=result,
            image=image,
            output_dir=output_dir,
            write_arrow=self.config.save_arrow,
            write_crops=self.config.save_crops,
            write_viz=self.config.save_viz,
            write_original=self.config.save_original,
        )

        return result

    def _filter_segments(self, segments: List[Segment]) -> List[Segment]:
        """Apply configured filters to segments."""
        return filter_segments(
            segments,
            min_area=self.config.min_area,
            max_area=self.config.max_area,
            min_score=self.config.min_score,
            labels=self.config.filter_labels,
        )

    def _print_summary(self) -> None:
        """Print processing summary."""
        print(f"\n{'='*70}")
        print(f"📊 SEGMENTATION PIPELINE SUMMARY")
        print(f"{'='*70}")
        print(f"Total images:      {self.stats['total_images']}")
        print(f"✓ Successful:      {self.stats['successful']}")
        print(f"✗ Failed:          {self.stats['failed']}")
        print(f"Total segments:    {self.stats['total_segments']}")
        print(
            f"Avg segments/img:  {self.stats['total_segments']/max(1, self.stats['successful']):.1f}"
        )
        print(f"Total time:        {self.stats['total_time']:.2f}s")
        print(
            f"Avg time/img:      {self.stats['total_time']/max(1, self.stats['successful']):.3f}s"
        )
        print(f"{'='*70}")

        if self.stats["errors"] and self.config.verbose:
            print(f"\n❌ Errors ({len(self.stats['errors'])}):")
            for rel_path, error in self.stats["errors"][:10]:  # Show first 10
                print(f"   {rel_path}: {error}")
            if len(self.stats["errors"]) > 10:
                print(f"   ... and {len(self.stats['errors']) - 10} more")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.model:
            self.model.cleanup()


# Convenience function
def run_pipeline(config: PipelineConfig, list_file: str) -> Dict[str, Any]:
    """Convenience function to run pipeline."""
    with SegmentationPipeline(config) as pipeline:
        return pipeline.run_on_image_list(list_file)

    return {}  # Should not reach here
