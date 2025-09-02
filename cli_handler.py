"""
CLI Handler Module for Telegram Sticker Animator

Provides comprehensive command-line argument parsing, validation, and input
processing functionality as specified in the requirements document.
"""

import argparse
import glob
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from config import (
    IMAGE_FORMATS,
    REMBG_MODELS,
    DEFAULT_REMBG_MODEL,
    TELEGRAM_MAX_DURATION,
    TELEGRAM_MAX_FILE_SIZE,
    ERROR_MESSAGES
)


# Input validation constants
VALID_INPUT_EXTENSIONS = ['.jpg', '.jpeg', '.png']
VALID_OUTPUT_EXTENSION = '.webm'

# Constraints
MAX_FPS = 60
MIN_FPS = 10
MAX_DURATION_PER_FRAME = 1.0
MIN_DURATION_PER_FRAME = 0.033  # ~30fps
MAX_TOTAL_DURATION = 3.0  # Telegram limit

# Default values
DEFAULTS = {
    'fps': 30,
    'duration': 0.1,
    'quality': 8,
    'remove_bg': True,
    'bg_model': 'u2net'
}

# Error messages
ERROR_MESSAGES_CLI = {
    'no_input': 'Error: Input file/directory not found: {}',
    'no_images': 'Error: No valid image files found in: {}',
    'invalid_extension': 'Error: Invalid file extension. Supported: {}',
    'output_exists': 'Warning: Output file exists and will be overwritten: {}',
    'ffmpeg_missing': 'Error: FFmpeg not found. Please install FFmpeg.',
    'rembg_missing': 'Warning: rembg not installed. Background removal disabled.',
    'file_too_large': 'Warning: Output file exceeds 256KB limit: {} bytes',
    'duration_exceeded': 'Error: Total duration exceeds 3 seconds: {} seconds'
}


class CLIHandler:
    """
    Enhanced CLI handler class providing all functionality specified 
    in the requirements document.
    """
    
    def __init__(self):
        """Initialize the CLI handler with argument parser."""
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
  python telegram_animator.py images/ output.webm --fps 25 --duration 0.2

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

Constraints:
  • Max file size: 256KB (Telegram limit)
  • Max duration: 3 seconds (Telegram limit)
  • FPS range: 10-60
  • Frame duration: 0.033-1.0 seconds
            """
        )
        
        # Positional arguments
        parser.add_argument('input', 
                          help='Input image(s) - file, directory, or pattern')
        parser.add_argument('output',
                          help='Output WebM file path')
        
        # Background removal
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
                          default=DEFAULTS['fps'],
                          help=f'Frames per second (default: {DEFAULTS["fps"]})')
        parser.add_argument('--duration',
                          type=float,
                          default=DEFAULTS['duration'],
                          help=f'Duration per image in seconds (default: {DEFAULTS["duration"]})')
        
        # Quality settings
        parser.add_argument('--quality',
                          type=int,
                          choices=range(1, 11),
                          default=DEFAULTS['quality'],
                          help=f'Output quality 1-10 (default: {DEFAULTS["quality"]})')
        
        # Advanced options
        parser.add_argument('--bg-model',
                          default=DEFAULTS['bg_model'],
                          choices=list(REMBG_MODELS.keys()),
                          help=f'Background removal model (default: {DEFAULTS["bg_model"]})')
        
        # Additional options
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
        
        Checks:
        - Input file/directory exists
        - Output directory is writable
        - Numeric ranges are valid
        - Total duration < 3 seconds
        
        Returns:
            True if validation passes, False otherwise
        """
        # Check input exists
        if not self._validate_input_exists(args.input):
            return False
        
        # Check output directory is writable  
        if not self._validate_output_writable(args.output, args.overwrite):
            return False
        
        # Validate numeric ranges
        if not self._validate_numeric_ranges(args):
            return False
        
        # Get image paths and check total duration
        image_paths = self.process_input_path(args.input)
        if not image_paths:
            print(ERROR_MESSAGES_CLI['no_images'].format(args.input))
            return False
        
        # Check total duration < 3 seconds
        total_duration = len(image_paths) * args.duration
        if total_duration > MAX_TOTAL_DURATION:
            print(ERROR_MESSAGES_CLI['duration_exceeded'].format(total_duration))
            return False
        
        return True
    
    def _validate_input_exists(self, input_path: str) -> bool:
        """Validate that input path exists and is accessible."""
        path = Path(input_path)
        
        # Handle wildcard patterns
        if '*' in input_path or '?' in input_path:
            matches = glob.glob(input_path)
            if not matches:
                print(f"No files found matching pattern: {input_path}")
                return False
            return True
        
        # Check if path exists
        if not path.exists():
            print(ERROR_MESSAGES_CLI['no_input'].format(input_path))
            return False
        
        return True
    
    def _validate_output_writable(self, output_path: str, overwrite: bool) -> bool:
        """Validate output path and handle overwrite logic."""
        path = Path(output_path)
        
        # Check output extension
        if path.suffix.lower() != VALID_OUTPUT_EXTENSION:
            print(f"Output must have {VALID_OUTPUT_EXTENSION} extension, got: {path.suffix}")
            return False
        
        # Check if output directory exists and is writable
        output_dir = path.parent
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(f"Cannot create output directory: {output_dir}")
                return False
        
        # Handle file overwrite
        if path.exists() and not overwrite:
            print(ERROR_MESSAGES_CLI['output_exists'].format(output_path))
            response = input("Overwrite? (y/n): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Operation cancelled")
                return False
        
        return True
    
    def _validate_numeric_ranges(self, args: argparse.Namespace) -> bool:
        """Validate that numeric arguments are within acceptable ranges."""
        # Validate FPS
        if not (MIN_FPS <= args.fps <= MAX_FPS):
            print(f"FPS must be between {MIN_FPS} and {MAX_FPS}, got: {args.fps}")
            return False
        
        # Validate duration per frame
        if not (MIN_DURATION_PER_FRAME <= args.duration <= MAX_DURATION_PER_FRAME):
            print(f"Duration must be between {MIN_DURATION_PER_FRAME} and {MAX_DURATION_PER_FRAME} seconds, got: {args.duration}")
            return False
        
        # Quality is validated by argparse choices constraint
        
        return True
    
    def process_input_path(self, input_path: str) -> List[str]:
        """
        Process input path and return sorted list of valid image files.
        
        - Accept single image file
        - Accept directory of images
        - Accept wildcard patterns (*.jpg)
        - Sort images alphabetically
        - Validate file extensions
        
        Args:
            input_path: Input path string
            
        Returns:
            Sorted list of valid image file paths
        """
        paths = []
        
        # Handle wildcard patterns
        if '*' in input_path or '?' in input_path:
            matches = glob.glob(input_path)
            for match in matches:
                if self._is_valid_image_extension(match):
                    paths.append(match)
        
        # Handle directory
        elif Path(input_path).is_dir():
            directory = Path(input_path)
            for ext in VALID_INPUT_EXTENSIONS:
                # Check both lowercase and uppercase extensions
                patterns = [f"*{ext}", f"*{ext.upper()}"]
                for pattern in patterns:
                    paths.extend(str(p) for p in directory.glob(pattern))
        
        # Handle single file
        elif Path(input_path).is_file():
            if self._is_valid_image_extension(input_path):
                paths.append(input_path)
        
        # Remove duplicates and sort alphabetically
        unique_paths = list(set(paths))
        unique_paths.sort()
        
        return unique_paths
    
    def _is_valid_image_extension(self, filepath: str) -> bool:
        """Check if file has a valid image extension."""
        path = Path(filepath)
        return path.suffix.lower() in VALID_INPUT_EXTENSIONS
    
    def get_image_paths(self, input_path: str) -> List[str]:
        """
        Alias for process_input_path for compatibility.
        
        This method provides the interface specified in the requirements
        while maintaining compatibility with the main implementation.
        """
        return self.process_input_path(input_path)
    
    def get_validation_summary(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Get a summary of validation results for reporting.
        
        Args:
            args: Parsed arguments
            
        Returns:
            Dictionary with validation information
        """
        image_paths = self.process_input_path(args.input)
        total_duration = len(image_paths) * args.duration
        
        return {
            'input_path': args.input,
            'output_path': args.output,
            'image_count': len(image_paths),
            'image_paths': image_paths,
            'total_duration': total_duration,
            'fps': args.fps,
            'quality': args.quality,
            'remove_bg': args.remove_bg,
            'bg_model': args.bg_model,
            'within_duration_limit': total_duration <= MAX_TOTAL_DURATION,
            'estimated_frames': len(image_paths) * int(args.fps * args.duration)
        }


def create_cli_handler() -> CLIHandler:
    """
    Factory function to create a CLI handler instance.
    
    Returns:
        Configured CLIHandler instance
    """
    return CLIHandler()


# Implementation example as specified in requirements
def main():
    """
    Example CLI implementation showing integration with processing modules.
    
    This demonstrates the complete flow from argument parsing to
    module initialization as specified in the requirements.
    """
    # Initialize CLI handler
    cli = CLIHandler()
    
    try:
        # Parse arguments
        args = cli.parse_args()
        
        # Validate inputs
        if not cli.validate_args(args):
            sys.exit(1)
        
        # Get validation summary
        summary = cli.get_validation_summary(args)
        
        print(f"Validation Summary:")
        print(f"  Input: {summary['input_path']}")
        print(f"  Output: {summary['output_path']}")
        print(f"  Images found: {summary['image_count']}")
        print(f"  Total duration: {summary['total_duration']:.2f}s")
        print(f"  Estimated frames: {summary['estimated_frames']}")
        print(f"  Within limits: {summary['within_duration_limit']}")
        
        if args.verbose:
            print(f"  Image files:")
            for i, path in enumerate(summary['image_paths'], 1):
                print(f"    {i}. {path}")
        
        # Initialize processing modules (placeholder for integration)
        print("✅ CLI validation successful - ready for processing")
        
        # Here would be the actual processing pipeline:
        # from image_processor import ImageProcessor
        # from bg_remover import BackgroundRemover  
        # from video_creator import VideoCreator
        # ... processing logic ...
        
    except Exception as e:
        print(f"CLI Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()
