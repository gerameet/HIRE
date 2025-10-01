#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from segmentation_pipeline.core.base import ModelConfig
from segmentation_pipeline.pipeline import PipelineConfig, SegmentationPipeline
from segmentation_pipeline.models import list_available_models


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="🎯 Modular Segmentation Pipeline - Process images with SOTA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with dummy model (testing)
  python run_segmentation.py --data_root ./data --list_file images.txt \\
    --output_root ./output --model dummy

  # Run with SAM
  python run_segmentation.py --data_root ./data --list_file images.txt \\
    --output_root ./output --model sam \\
    --checkpoint sam_vit_b_01ec64.pth --sam_model_type vit_b

  # Run with YOLOv8 segmentation
  python run_segmentation.py --data_root ./data --list_file images.txt \\
    --output_root ./output --model yolo \\
    --checkpoint yolov8n-seg.pt --min_score 0.5 --save_viz

  # Run with Mask2Former (semantic segmentation)
  python run_segmentation.py --data_root ./data --list_file images.txt \\
    --output_root ./output --model mask2former \\
    --checkpoint facebook/mask2former-swin-base-ade-semantic \\
    --mask2former_task semantic

  # Run with CLIPSeg (text-prompted)
  python run_segmentation.py --data_root ./data --list_file images.txt \\
    --output_root ./output --model clipseg \\
    --text_prompts "person" "car" "building"
        """
    )
    
    # Required arguments
    required = parser.add_argument_group("Required arguments")
    required.add_argument(
        "--data_root",
        required=True,
        help="Root directory containing images"
    )
    required.add_argument(
        "--list_file",
        required=True,
        help="Text file with image paths (one per line, relative to data_root)"
    )
    required.add_argument(
        "--output_root",
        required=True,
        help="Directory to save results"
    )
    
    # Model selection
    model_group = parser.add_argument_group("Model selection")
    model_group.add_argument(
        "--model",
        default="dummy",
        choices=["dummy", "sam", "mask2former", "segformer", "yolo", "detectron2", "clipseg"],
        help="Segmentation model to use"
    )
    model_group.add_argument(
        "--checkpoint",
        help="Path to model checkpoint or HuggingFace model ID"
    )
    model_group.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps"],
        help="Device to run model on (auto-detect if not specified)"
    )
    
    # Model-specific options
    sam_group = parser.add_argument_group("SAM options")
    sam_group.add_argument(
        "--sam_model_type",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model architecture"
    )
    sam_group.add_argument(
        "--sam_automatic",
        action="store_true",
        default=True,
        help="Use automatic mask generation (default: True)"
    )
    
    mask2former_group = parser.add_argument_group("Mask2Former options")
    mask2former_group.add_argument(
        "--mask2former_task",
        default="instance",
        choices=["instance", "semantic", "panoptic"],
        help="Mask2Former task type"
    )
    
    clipseg_group = parser.add_argument_group("CLIPSeg options")
    clipseg_group.add_argument(
        "--text_prompts",
        nargs="+",
        help="Text prompts for CLIPSeg segmentation"
    )
    
    # Processing options
    processing = parser.add_argument_group("Processing options")
    processing.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    processing.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    
    # Filtering options
    filtering = parser.add_argument_group("Filtering options")
    filtering.add_argument(
        "--min_area",
        type=int,
        default=0,
        help="Minimum segment area in pixels (default: 0)"
    )
    filtering.add_argument(
        "--max_area",
        type=int,
        help="Maximum segment area in pixels"
    )
    filtering.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Minimum confidence score (default: 0.0)"
    )
    filtering.add_argument(
        "--filter_labels",
        nargs="+",
        help="Only keep segments with these labels"
    )
    filtering.add_argument(
        "--apply_nms",
        action="store_true",
        help="Apply Non-Maximum Suppression to remove overlapping segments"
    )
    filtering.add_argument(
        "--nms_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS (default: 0.5)"
    )
    
    # Output options
    output = parser.add_argument_group("Output options")
    output.add_argument(
        "--save_arrow",
        action="store_true",
        default=True,
        help="Save segments in Arrow format (default: True)"
    )
    output.add_argument(
        "--save_crops",
        action="store_true",
        help="Save individual instance crops"
    )
    output.add_argument(
        "--save_viz",
        action="store_true",
        help="Save visualization with overlays"
    )
    output.add_argument(
        "--save_original",
        action="store_true",
        help="Copy original image to output directory"
    )
    
    # Display options
    display = parser.add_argument_group("Display options")
    display.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output"
    )
    display.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar"
    )
    display.add_argument(
        "--list_models",
        action="store_true",
        help="List available models and exit"
    )
    
    return parser


def main(argv=None):
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # List models and exit
    if args.list_models:
        print("📋 Available segmentation models:")
        for model_name in list_available_models():
            print(f"  • {model_name}")
        return 0
    
    # Print banner
    if args.verbose:
        print("=" * 70)
        print("🎯 MODULAR SEGMENTATION PIPELINE")
        print("=" * 70)
    
    # Validate inputs
    if not Path(args.data_root).exists():
        print(f"❌ Error: data_root does not exist: {args.data_root}")
        return 1
    
    if not Path(args.list_file).exists():
        print(f"❌ Error: list_file does not exist: {args.list_file}")
        return 1
    
    # Create model config
    extra_params = {}
    
    if args.model == "dummy":
        extra_params["num_masks"] = 3
    
    elif args.model == "sam":
        if not args.checkpoint:
            print("❌ Error: --checkpoint required for SAM model")
            print("   Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            return 1
        extra_params["use_automatic"] = args.sam_automatic
    
    elif args.model == "mask2former":
        extra_params["task"] = args.mask2former_task
    
    elif args.model == "clipseg":
        if not args.text_prompts:
            print("❌ Error: --text_prompts required for CLIPSeg model")
            return 1
        extra_params["text_prompts"] = args.text_prompts
    
    model_config = ModelConfig(
        device=args.device,
        checkpoint=args.checkpoint,
        model_type=getattr(args, "sam_model_type", None),
        batch_size=args.batch_size,
        confidence_threshold=args.min_score,
        min_area=args.min_area,
        extra_params=extra_params
    )
    
    # Create pipeline config
    pipeline_config = PipelineConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        model_name=args.model,
        model_config=model_config,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        min_area=args.min_area,
        max_area=args.max_area,
        min_score=args.min_score,
        filter_labels=args.filter_labels,
        apply_nms=args.apply_nms,
        nms_threshold=args.nms_threshold,
        save_arrow=args.save_arrow,
        save_crops=args.save_crops,
        save_viz=args.save_viz,
        save_original=args.save_original,
        show_progress=not args.no_progress,
        verbose=args.verbose
    )
    
    # Run pipeline
    try:
        with SegmentationPipeline(pipeline_config) as pipeline:
            results = pipeline.run_on_image_list(args.list_file)
        
        # Check for failures
        if results["stats"]["failed"] > 0:
            print(f"\n⚠️  Warning: {results['stats']['failed']} images failed to process")
            return 1
        
        print("\n✅ Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())