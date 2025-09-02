"""
CLI Demo for Video Creation Integration

Demonstrates complete pipeline: ImageProcessor -> BackgroundRemover -> VideoCreator
Creates WebM animations optimized for Telegram's 256KB file size limit.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from image_processor import ImageProcessor
from background_remover import BackgroundRemover
from video_creator import TelegramWebMCreator, create_telegram_video
from config import (
    REMBG_MODELS, 
    DEFAULT_REMBG_MODEL,
    TELEGRAM_MAX_FILE_SIZE,
    TELEGRAM_MAX_DURATION,
    TELEGRAM_RECOMMENDED_FPS,
    VP9_QUALITY_LEVELS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_cli_parser():
    """
    Create CLI argument parser for video creation pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Telegram Sticker Animator - Image to WebM Video Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Background Removal Models:
{chr(10).join(f'  {k}: {v}' for k, v in REMBG_MODELS.items())}

Quality Levels (1-10):
  1:  Lowest quality, smallest file
  5:  Balanced quality/size
  8:  Default quality (recommended)
  10: Highest quality, largest file

Examples:
  # Create video from multiple images
  python cli_video_demo.py image1.jpg image2.png image3.jpg -o animation.webm
  
  # Create video without background removal
  python cli_video_demo.py *.jpg --no-remove-bg -o simple_animation.webm
  
  # Create video with custom timing and quality
  python cli_video_demo.py images/*.png --fps 20 --duration 0.2 --quality 6
  
  # Create video with specific background removal model
  python cli_video_demo.py photos/*.jpg --bg-model u2netp --verbose
        """
    )
    
    # Input/output arguments
    parser.add_argument(
        'images', 
        nargs='+', 
        help='Input image files (supports wildcards)'
    )
    
    parser.add_argument(
        '-o', '--output', 
        default='telegram_animation.webm',
        help='Output WebM file path (default: telegram_animation.webm)'
    )
    
    # Video creation arguments
    parser.add_argument(
        '--fps', 
        type=int,
        default=TELEGRAM_RECOMMENDED_FPS,
        help=f'Frames per second (default: {TELEGRAM_RECOMMENDED_FPS})'
    )
    
    parser.add_argument(
        '--duration', 
        type=float,
        default=0.1,
        help='Duration to display each image in seconds (default: 0.1)'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        choices=list(VP9_QUALITY_LEVELS.keys()),
        default=8,
        help='Video quality level 1-10 (default: 8)'
    )
    
    # Background removal arguments
    bg_group = parser.add_mutually_exclusive_group()
    bg_group.add_argument(
        '--remove-bg',
        action='store_true',
        default=True,
        help='Enable background removal (default)'
    )
    bg_group.add_argument(
        '--no-remove-bg',
        action='store_true', 
        help='Disable background removal'
    )
    
    parser.add_argument(
        '--bg-model',
        choices=list(REMBG_MODELS.keys()),
        default=DEFAULT_REMBG_MODEL,
        help=f'Background removal model (default: {DEFAULT_REMBG_MODEL})'
    )
    
    # Processing options
    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        default=[512, 512],
        help='Output video dimensions (default: 512 512)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output file'
    )
    
    # Debug/info options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with detailed processing info'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without creating video'
    )
    
    return parser

def validate_args(args):
    """
    Validate CLI arguments and check constraints.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        True if valid, False otherwise
    """
    # Check image files exist
    valid_images = []
    for image_path in args.images:
        path = Path(image_path)
        if path.exists() and path.is_file():
            valid_images.append(str(path))
        else:
            print(f"❌ Image file not found: {image_path}")
    
    if not valid_images:
        print("❌ No valid image files found")
        return False
    
    args.images = valid_images
    
    # Check output file
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"❌ Output file already exists: {args.output}")
        print("   Use --overwrite to replace existing file")
        return False
    
    # Validate video duration constraints
    total_duration = len(args.images) * args.duration
    if total_duration > TELEGRAM_MAX_DURATION:
        print(f"⚠️  Warning: Total duration ({total_duration:.1f}s) exceeds Telegram limit ({TELEGRAM_MAX_DURATION}s)")
        print("   Video may be automatically optimized to fit constraints")
    
    # Validate FPS
    if args.fps <= 0 or args.fps > 60:
        print("❌ FPS must be between 1 and 60")
        return False
    
    # Validate duration per frame
    if args.duration <= 0:
        print("❌ Duration per frame must be positive")
        return False
    
    return True

def setup_logging(verbose: bool):
    """Configure logging based on verbosity level."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Also enable debug logging for our modules
        logging.getLogger('image_processor').setLevel(logging.DEBUG)
        logging.getLogger('background_remover').setLevel(logging.DEBUG) 
        logging.getLogger('video_creator').setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

def print_processing_summary(args, bg_remover, processor):
    """Print summary of processing settings."""
    print("🎬 Telegram Sticker Animator")
    print("=" * 50)
    
    print(f"\n📂 Input:")
    print(f"   Images: {len(args.images)} files")
    for i, img in enumerate(args.images[:5], 1):  # Show first 5
        print(f"     {i}. {Path(img).name}")
    if len(args.images) > 5:
        print(f"     ... and {len(args.images) - 5} more")
    
    print(f"\n🎥 Video Settings:")
    print(f"   Output: {args.output}")
    print(f"   Dimensions: {args.size[0]}×{args.size[1]}")
    print(f"   FPS: {args.fps}")
    print(f"   Duration per frame: {args.duration}s")
    print(f"   Total duration: {len(args.images) * args.duration:.1f}s")
    print(f"   Quality level: {args.quality}")
    
    # Background removal info
    bg_info = bg_remover.get_model_info()
    print(f"\n🎨 Background Removal:")
    print(f"   Enabled: {bg_info['enabled']}")
    if bg_info['enabled']:
        print(f"   Model: {bg_info['model']} - {bg_info['model_description']}")
        print(f"   Available: {bg_info['available']}")
    
    # Telegram constraints
    print(f"\n📏 Telegram Constraints:")
    print(f"   Max file size: {TELEGRAM_MAX_FILE_SIZE/1024:.0f} KB")
    print(f"   Max duration: {TELEGRAM_MAX_DURATION}s")
    print(f"   Automatic optimization: Enabled")

def main():
    """Main CLI function for video creation pipeline."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    if not validate_args(args):
        return 1
    
    # Initialize components
    try:
        # Background remover
        remove_bg_enabled = not args.no_remove_bg
        bg_remover = BackgroundRemover(enabled=remove_bg_enabled, model=args.bg_model)
        
        # Image processor
        processor = ImageProcessor(target_size=tuple(args.size))
        
        # Print processing summary
        print_processing_summary(args, bg_remover, processor)
        
        if args.dry_run:
            print("\n🔍 Dry run complete - no video created")
            return 0
        
        print(f"\n🔄 Processing {len(args.images)} images...")
        
        # Process images
        processed_frames = []
        
        for i, image_path in enumerate(args.images, 1):
            try:
                if args.verbose:
                    print(f"\n   Processing image {i}/{len(args.images)}: {Path(image_path).name}")
                else:
                    print(f"   [{i}/{len(args.images)}] {Path(image_path).name}")
                
                # Load and process image
                processed_frame = processor.process_image(image_path)
                
                # Apply background removal if enabled
                if remove_bg_enabled:
                    from PIL import Image
                    # Convert numpy back to PIL for background removal
                    pil_image = Image.fromarray(processed_frame, 'RGBA')
                    bg_removed = bg_remover.process(pil_image)
                    # Convert back to numpy
                    processed_frame = np.array(bg_removed)
                
                processed_frames.append(processed_frame)
                
                if args.verbose:
                    print(f"     Shape: {processed_frame.shape}, dtype: {processed_frame.dtype}")
                
            except Exception as e:
                print(f"   ❌ Error processing {image_path}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        if not processed_frames:
            print("❌ No images could be processed")
            return 1
        
        print(f"\n✅ Successfully processed {len(processed_frames)} images")
        
        # Create video
        print(f"\n🎬 Creating WebM video...")
        print(f"   Using VP9 codec with quality level {args.quality}")
        
        # Initialize video creator
        video_creator = TelegramWebMCreator()
        
        # Create video with size optimization
        success = video_creator.create_video(
            processed_frames,
            args.output,
            fps=args.fps,
            duration_per_frame=args.duration
        )
        
        if success:
            # Check final result
            output_path = Path(args.output)
            if output_path.exists():
                file_size = output_path.stat().st_size
                duration = len(processed_frames) * args.duration
                
                print(f"\n🎉 Video created successfully!")
                print(f"   File: {args.output}")
                print(f"   Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                print(f"   Duration: {duration:.1f}s")
                print(f"   Frames: {len(processed_frames)}")
                print(f"   Resolution: {args.size[0]}×{args.size[1]}")
                print(f"   FPS: {args.fps}")
                
                # Telegram compatibility check
                if file_size <= TELEGRAM_MAX_FILE_SIZE:
                    print(f"   ✅ Telegram compatible (≤ {TELEGRAM_MAX_FILE_SIZE/1024:.0f} KB)")
                else:
                    print(f"   ⚠️  Exceeds Telegram limit ({TELEGRAM_MAX_FILE_SIZE/1024:.0f} KB)")
                
                if duration <= TELEGRAM_MAX_DURATION:
                    print(f"   ✅ Duration compatible (≤ {TELEGRAM_MAX_DURATION}s)")
                else:
                    print(f"   ⚠️  Exceeds Telegram duration limit ({TELEGRAM_MAX_DURATION}s)")
                
            else:
                print("❌ Video file was not created")
                return 1
                
        else:
            print("❌ Video creation failed")
            print("   Check that FFmpeg is installed and accessible")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Processing cancelled by user")
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues if not available
    exit(main())
