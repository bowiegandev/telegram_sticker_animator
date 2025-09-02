# CLI Interface Module

## Implementation Goal
Create a user-friendly command-line interface for the Telegram animator script with flexible input/output options and configuration parameters.

## Technical Details

### Argument Parsing
```python
import argparse

def create_parser() -> argparse.ArgumentParser:
    """
    Setup command-line argument parser with:
    - Positional arguments for input/output
    - Optional flags for configuration
    - Help documentation
    """
```

### Input Handling
```python
def process_input_path(input_path: str) -> List[str]:
    """
    - Accept single image file
    - Accept directory of images
    - Accept wildcard patterns (*.jpg)
    - Sort images alphabetically
    - Validate file extensions
    """
```

## Code Structure

```python
import argparse
import sys
from pathlib import Path
from typing import List, Optional

class CLIHandler:
    def __init__(self):
        self.parser = self.create_parser()
        
    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog='telegram_animator',
            description='Convert JPG/JPEG images to WebM animations for Telegram stickers'
        )
        
        # Positional arguments
        parser.add_argument('input', 
                          help='Input image(s) - file, directory, or pattern')
        parser.add_argument('output',
                          help='Output WebM file path')
        
        # Background removal
        parser.add_argument('--remove-bg', 
                          action='store_true',
                          default=True,
                          help='Enable background removal (default)')
        parser.add_argument('--no-remove-bg',
                          dest='remove_bg',
                          action='store_false',
                          help='Disable background removal')
        
        # Animation settings
        parser.add_argument('--fps',
                          type=int,
                          default=30,
                          help='Frames per second (default: 30)')
        parser.add_argument('--duration',
                          type=float,
                          default=0.1,
                          help='Duration per image in seconds (default: 0.1)')
        
        # Quality settings
        parser.add_argument('--quality',
                          type=int,
                          choices=range(1, 11),
                          default=8,
                          help='Output quality 1-10 (default: 8)')
        
        # Advanced options
        parser.add_argument('--bg-model',
                          default='u2net',
                          choices=['u2net', 'u2netp', 'u2net_human_seg'],
                          help='Background removal model (default: u2net)')
        
        return parser
    
    def validate_args(self, args) -> bool:
        """Validate parsed arguments"""
        # Check input exists
        # Check output directory is writable
        # Validate numeric ranges
        # Check total duration < 3 seconds
```

## Integration Points
- **Input**: Command-line arguments from sys.argv
- **Output**: Parsed configuration dictionary
- **Modules**: Calls image_processor, bg_remover, video_creator

## Key Parameters

```python
# Input validation
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
```

## Main Script Flow

```python
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
    
    # Validate
    if not cli.validate_args(args):
        sys.exit(1)
    
    # Get image paths
    image_paths = cli.get_image_paths(args.input)
    if not image_paths:
        print("No valid images found")
        sys.exit(1)
    
    # Initialize processors
    img_processor = ImageProcessor()
    bg_remover = BackgroundRemover(enabled=args.remove_bg, model=args.bg_model)
    video_creator = VideoCreator(fps=args.fps, quality=args.quality)
    
    # Process images
    processed_frames = []
    for path in image_paths:
        frame = img_processor.process(path)
        if args.remove_bg:
            frame = bg_remover.process(frame)
        processed_frames.append(frame)
    
    # Create video
    success = video_creator.create(
        processed_frames,
        args.output,
        duration_per_frame=args.duration
    )
    
    if success:
        print(f"✓ Created {args.output}")
    else:
        print("✗ Failed to create video")
        sys.exit(1)
```

## Usage Examples

```bash
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
```

## Error Messages

```python
ERROR_MESSAGES = {
    'no_input': 'Error: Input file/directory not found: {}',
    'no_images': 'Error: No valid image files found in: {}',
    'invalid_extension': 'Error: Invalid file extension. Supported: {}',
    'output_exists': 'Warning: Output file exists and will be overwritten: {}',
    'ffmpeg_missing': 'Error: FFmpeg not found. Please install FFmpeg.',
    'rembg_missing': 'Warning: rembg not installed. Background removal disabled.',
    'file_too_large': 'Warning: Output file exceeds 256KB limit: {} bytes',
    'duration_exceeded': 'Error: Total duration exceeds 3 seconds: {} seconds'
}
```

## Implementation Example

```python
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List

def main():
    parser = argparse.ArgumentParser(
        description='Convert images to Telegram animated stickers'
    )
    
    # Add arguments
    parser.add_argument('input', help='Input image(s)')
    parser.add_argument('output', help='Output WebM file')
    parser.add_argument('--remove-bg', action='store_true', default=True)
    parser.add_argument('--no-remove-bg', dest='remove_bg', action='store_false')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument('--quality', type=int, default=8, choices=range(1, 11))
    
    args = parser.parse_args()
    
    # Process based on arguments
    try:
        from image_processor import ImageProcessor
        from bg_remover import BackgroundRemover
        from video_creator import VideoCreator
        
        # Initialize and run
        processor = ImageProcessor()
        remover = BackgroundRemover(enabled=args.remove_bg)
        creator = VideoCreator()
        
        # ... processing logic ...
        
        print(f"Successfully created: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
