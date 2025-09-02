#!/usr/bin/env python3
"""
Telegram Sticker Animator - Main CLI Entry Point

Convert JPG/JPEG images to WebM animations for Telegram stickers with optional
background removal and comprehensive configuration options.
"""

import argparse
import sys
import os
import glob
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from image_processor import ImageProcessor
from background_remover import BackgroundRemover
from video_creator import TelegramWebMCreator
from config import (
    IMAGE_FORMATS,
    REMBG_MODELS,
    DEFAULT_REMBG_MODEL,
    VP9_QUALITY_LEVELS,
    TELEGRAM_MAX_DURATION,
    TELEGRAM_MAX_FILE_SIZE,
    ERROR_MESSAGES,
    get_config
)


class CLIHandler:
    """
    Command-line interface handler for Telegram animator script.
    
    Provides flexible input/output options and configuration parameters
    as specified in the requirements.
    """
    
    def __init__(self):
        self.parser = self.create_parser()
        self.logger = logging.getLogger(__name__)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """
        Setup command-line argument parser with:
        - Positional arguments for input/output
        - Optional flags for configuration
        - Help documentation
        """
        parser = argparse.ArgumentParser(
            prog='telegram_animator',
            description='Convert JPG/JPEG images to WebM animations for Telegram stickers',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f"""
Examples:
  # Basic usage with single image
  python telegram_animator.py image.jpg output.webm

  # Process directory of images
  python telegram_animator.py ./images/ animation.webm

  # Disable background removal
  python telegram_animator.py images/ output.webm --no-remove-bg

  # Custom animation settings
  python telegram_animator.py images/ output.webv --fps 25 --duration 0.2

  # Low quality for smaller file size
  python telegram_animator.py images/ output.webm --quality 3

  # Use specific background removal model
  python telegram_animator.py person.jpg output.webm --bg-model u2net_human_seg

  # Process with wildcard pattern
  python telegram_animator.py "*.jpg" output.webm

Background Removal Models:
{chr(10).join(f'  {k}: {v}' for k, v in REMBG_MODELS.items())}

Quality Levels (1-10):
  1-3: Low quality, small file size
  4-6: Balanced quality and size
  7-8: High quality (default: 8)
  9-10: Maximum quality, larger files
            """
        )
        
        # Positional arguments
        parser.add_argument('input',
                          help='Input image(s) - file, directory, or pattern')
        parser.add_argument('output',
                          help='Output WebM file path')
        
        # Background removal options
        bg_group = parser.add_mutually_exclusive_group()
        bg_group.add_argument('--remove-bg',
                            action='store_true',
                            default=True,
                            help='Enable background removal (default)')
        bg_group.add_argument('--no-remove-bg',
                            dest='remove_bg',
                            action='store_false',
                            help='Disable background removal')
        
        # Animation settings
        parser.add_argument('--fps',
                          type=int,
                          default=30,
                          metavar='N',
                          help='Frames per second (default: 30)')
        parser.add_argument('--duration',
                          type=float,
                          default=0.1,
                          metavar='SECONDS',
                          help='Duration per image in seconds (default: 0.1)')
        
        # Quality settings
        parser.add_argument('--quality',
                          type=int,
                          choices=range(1, 11),
                          default=8,
                          metavar='N',
                          help='Output quality 1-10 (default: 8)')
        
        # Advanced options
        parser.add_argument('--bg-model',
                          default=DEFAULT_REMBG_MODEL,
                          choices=list(REMBG_MODELS.keys()),
                          help=f'Background removal model (default: {DEFAULT_REMBG_MODEL})')
        
        # Output options
        parser.add_argument('--overwrite',
                          action='store_true',
                          help='Overwrite output file if it exists')
        parser.add_argument('--verbose', '-v',
                          action='store_true',
                          help='Enable verbose output')
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        return self.parser.parse_args(args)
    
    def validate_args(self, args: argparse.Namespace) -> bool:
        """
        Validate parsed arguments.
        
        Returns:
            True if validation passes, False otherwise
        """
        # Check input exists
        if not self._validate_input_path(args.input):
            return False
        
        # Check output directory is writable
        if not self._validate_output_path(args.output, args.overwrite):
            return False
        
        # Validate numeric ranges
        if not self._validate_numeric_args(args):
            return False
        
        # Get image paths for duration check
        image_paths = self.get_image_paths(args.input)
        if not image_paths:
            print("❌ No valid images found")
            return False
        
        # Check total duration < 3 seconds
        total_duration = len(image_paths) * args.duration
        if total_duration > TELEGRAM_MAX_DURATION:
            print(f"❌ Total duration ({total_duration:.2f}s) exceeds Telegram limit ({TELEGRAM_MAX_DURATION}s)")
            print(f"   Try reducing --duration or number of images")
            return False
        
        return True
    
    def _validate_input_path(self, input_path: str) -> bool:
        """Validate input path exists and is accessible."""
        path = Path(input_path)
        
        # Check if it's a wildcard pattern
        if '*' in input_path or '?' in input_path:
            matches = glob.glob(input_path)
            if not matches:
                print(f"❌ No files found matching pattern: {input_path}")
                return False
            return True
        
        # Check if path exists
        if not path.exists():
            print(f"❌ Input path not found: {input_path}")
            return False
        
        return True
    
    def _validate_output_path(self, output_path: str, overwrite: bool) -> bool:
        """Validate output path and handle overwrite logic."""
        path = Path(output_path)
        
        # Check output extension
        if path.suffix.lower() != '.webm':
            print(f"❌ Output file must have .webm extension, got: {path.suffix}")
            return False
        
        # Check if output directory exists and is writable
        output_dir = path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(f"❌ Cannot create output directory: {output_dir}")
                return False
        
        # Check if output file exists
        if path.exists() and not overwrite:
            print(f"⚠️  Output file exists: {output_path}")
            response = input("Overwrite? (y/n): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Operation cancelled")
                return False
        
        return True
    
    def _validate_numeric_args(self, args: argparse.Namespace) -> bool:
        """Validate numeric argument ranges."""
        # Validate FPS
        if not (10 <= args.fps <= 60):
            print(f"❌ FPS must be between 10 and 60, got: {args.fps}")
            return False
        
        # Validate duration per frame
        if not (0.033 <= args.duration <= 1.0):
            print(f"❌ Duration must be between 0.033 and 1.0 seconds, got: {args.duration}")
            return False
        
        # Quality is already validated by argparse choices
        
        return True
    
    def get_image_paths(self, input_path: str) -> List[str]:
        """
        Get sorted list of image paths from input.
        
        - Accept single image file
        - Accept directory of images  
        - Accept wildcard patterns (*.jpg)
        - Sort images alphabetically
        - Validate file extensions
        """
        paths = []
        
        # Handle wildcard patterns
        if '*' in input_path or '?' in input_path:
            matches = glob.glob(input_path)
            for match in matches:
                if self._is_valid_image_file(match):
                    paths.append(match)
        
        # Handle directory
        elif Path(input_path).is_dir():
            directory = Path(input_path)
            for ext in IMAGE_FORMATS:
                # Check both lowercase and uppercase extensions
                for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                    paths.extend(str(p) for p in directory.glob(pattern))
        
        # Handle single file
        elif Path(input_path).is_file():
            if self._is_valid_image_file(input_path):
                paths.append(input_path)
        
        # Remove duplicates and sort alphabetically
        unique_paths = list(set(paths))
        unique_paths.sort()
        
        return unique_paths
    
    def _is_valid_image_file(self, filepath: str) -> bool:
        """Check if file has valid image extension."""
        path = Path(filepath)
        return path.suffix.lower() in IMAGE_FORMATS
    
    def process_input_path(self, input_path: str) -> List[str]:
        """
        Process input path and return list of valid image files.
        This is an alias for get_image_paths for compatibility with requirements.
        """
        return self.get_image_paths(input_path)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    """
    Main entry point:
    1. Parse arguments
    2. Validate inputs  
    3. Initialize modules
    4. Process images
    5. Create video
    6. Verify output
    """
    # Parse CLI arguments
    cli = CLIHandler()
    args = cli.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not cli.validate_args(args):
        sys.exit(1)
    
    # Get image paths
    image_paths = cli.get_image_paths(args.input)
    if not image_paths:
        print("❌ No valid images found")
        sys.exit(1)
    
    print(f"📂 Found {len(image_paths)} image(s) to process")
    if args.verbose:
        for i, path in enumerate(image_paths, 1):
            print(f"  {i}. {path}")
    
    try:
        # Initialize processors
        print("🔧 Initializing processors...")
        
        img_processor = ImageProcessor()
        bg_remover = BackgroundRemover(enabled=args.remove_bg, model=args.bg_model)
        video_creator = TelegramWebMCreator()
        
        if args.verbose:
            bg_info = bg_remover.get_model_info()
            print(f"   Background removal: {'enabled' if bg_info['enabled'] else 'disabled'}")
            if bg_info['enabled']:
                print(f"   Model: {bg_info['model']} ({'available' if bg_info['available'] else 'fallback'})")
        
        # Process images
        print("🎨 Processing images...")
        processed_frames = []
        
        for i, path in enumerate(image_paths, 1):
            print(f"   Processing {i}/{len(image_paths)}: {Path(path).name}")
            
            # Load and process image
            frame = img_processor.process_image(path)
            
            # Apply background removal if enabled
            if args.remove_bg:
                from PIL import Image
                pil_image = Image.fromarray(frame)
                processed_pil = bg_remover.process(pil_image)
                frame = processed_pil if processed_pil else frame
                # Convert back to numpy if needed
                if hasattr(frame, 'save'):  # It's a PIL image
                    import numpy as np
                    frame = np.array(frame)
            
            processed_frames.append(frame)
        
        print(f"✅ Processed {len(processed_frames)} frames")
        
        # Create video
        print("🎬 Creating video...")
        success = video_creator.create_video(
            processed_frames,
            args.output,
            fps=args.fps,
            duration_per_frame=args.duration
        )
        
        if success:
            # Verify output and report statistics
            output_path = Path(args.output)
            if output_path.exists():
                file_size = output_path.stat().st_size
                duration = len(image_paths) * args.duration
                
                print(f"✅ Video created successfully!")
                print(f"📁 Output: {args.output}")
                print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                print(f"⏱️  Duration: {duration:.1f} seconds")
                print(f"🎞️  Frames: {len(processed_frames)} images × {args.fps} fps")
                
                # Check Telegram compatibility
                if file_size <= TELEGRAM_MAX_FILE_SIZE:
                    print("✅ File size within Telegram limit")
                else:
                    print(f"⚠️  File size exceeds Telegram limit ({TELEGRAM_MAX_FILE_SIZE/1024:.0f} KB)")
                
                if duration <= TELEGRAM_MAX_DURATION:
                    print("✅ Duration within Telegram limit")
                else:
                    print(f"⚠️  Duration exceeds Telegram limit ({TELEGRAM_MAX_DURATION}s)")
            else:
                print("❌ Output file was not created")
                sys.exit(1)
        else:
            print("❌ Failed to create video")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
