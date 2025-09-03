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
from cli_handler import CLIHandler
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
        
        # Configure advanced features if available
        if hasattr(args, 'interpolate') and args.interpolate:
            print(f"🎭 Enabling {args.interpolate} interpolation with {getattr(args, 'interp_frames', 2)} intermediate frames")
            video_creator.enable_interpolation(args.interpolate, getattr(args, 'interp_frames', 2))
        
        if hasattr(args, 'transition') and args.transition:
            transition_duration = getattr(args, 'transition_duration', 0.3)
            print(f"🔄 Enabling {args.transition} transitions with {transition_duration}s duration")
            
            # Get transition-specific parameters
            transition_kwargs = {}
            if args.transition == 'slide':
                transition_kwargs['direction'] = getattr(args, 'slide_direction', 'left')
            elif args.transition == 'scale':
                transition_kwargs['scale_type'] = getattr(args, 'scale_type', 'zoom_in')
            
            video_creator.enable_transitions(args.transition, transition_duration, **transition_kwargs)
        
        if hasattr(args, 'motion_blur') and getattr(args, 'motion_blur', 0) > 0:
            print(f"💫 Enabling motion blur with intensity {args.motion_blur}")
            video_creator.set_motion_blur(args.motion_blur)
        
        # Configure rotation animation
        if hasattr(args, 'rotation') and args.rotation:
            rotation_duration = getattr(args, 'rotation_duration', 2.0)
            rotation_steps = getattr(args, 'rotation_steps', 36)
            print(f"🔄 Enabling {args.rotation} rotation: {rotation_duration}s duration, {rotation_steps} steps")
            video_creator.enable_rotation(args.rotation, rotation_duration, rotation_steps)
        
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
