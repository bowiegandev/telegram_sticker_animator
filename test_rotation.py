#!/usr/bin/env python3
"""
Test script for the new 360-degree rotation animation feature.

This script creates a test image and demonstrates the rotation functionality.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from image_processor import ImageProcessor
from video_creator import TelegramWebMCreator


def create_test_image(output_path: str) -> str:
    """
    Create a test image with distinctive features for rotation testing.
    
    Args:
        output_path: Path to save the test image
        
    Returns:
        Path to the created test image
    """
    # Create a 512x512 image with a distinctive pattern
    img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a colorful arrow pointing right to show rotation
    # Arrow body (rectangle)
    draw.rectangle([150, 230, 300, 280], fill=(255, 100, 100, 255))
    
    # Arrow head (triangle)
    arrow_points = [(300, 230), (380, 255), (300, 280)]
    draw.polygon(arrow_points, fill=(255, 100, 100, 255))
    
    # Add some circles for additional visual reference
    draw.ellipse([200, 180, 250, 230], fill=(100, 255, 100, 255))  # Green circle top
    draw.ellipse([200, 280, 250, 330], fill=(100, 100, 255, 255))  # Blue circle bottom
    
    # Add center dot
    draw.ellipse([252, 252, 260, 260], fill=(255, 255, 0, 255))  # Yellow center dot
    
    # Try to add text (fallback to no text if font not available)
    try:
        # Try to use a default font
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    if font:
        draw.text((200, 350), "TEST", fill=(255, 255, 255, 255), font=font, anchor="mm")
    
    # Save the test image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"✅ Created test image: {output_path}")
    
    return output_path


def test_rotation_animation():
    """Test the rotation animation feature."""
    print("🧪 Testing 360-degree rotation animation feature")
    print("=" * 50)
    
    # Create test image
    test_image_path = "output/test_rotation_arrow.png"
    create_test_image(test_image_path)
    
    # Initialize image processor
    img_processor = ImageProcessor()
    
    # Process the test image
    print("🎨 Processing test image...")
    processed_frame = img_processor.process_image(test_image_path)
    print(f"   Image processed: {processed_frame.shape}")
    
    # Create video creator and enable rotation
    print("🔄 Setting up rotation animation...")
    video_creator = TelegramWebMCreator()
    
    # Test clockwise rotation
    video_creator.enable_rotation('clockwise', duration=2.0, steps=24)  # 24 steps for faster test
    
    # Create the rotating video
    print("🎬 Creating rotating video...")
    output_path = "output/test_rotation_clockwise.webm"
    
    success = video_creator.create_video(
        [processed_frame],  # Single image for rotation
        output_path,
        fps=30,
        duration_per_frame=2.0  # This will be overridden by rotation settings
    )
    
    if success:
        file_size = Path(output_path).stat().st_size
        print(f"✅ Clockwise rotation video created!")
        print(f"   Output: {output_path}")
        print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        # Test counterclockwise rotation
        print("\n🔄 Testing counterclockwise rotation...")
        video_creator_ccw = TelegramWebMCreator()
        video_creator_ccw.enable_rotation('counterclockwise', duration=1.5, steps=18)
        
        output_path_ccw = "output/test_rotation_counterclockwise.webm"
        success_ccw = video_creator_ccw.create_video(
            [processed_frame],
            output_path_ccw,
            fps=30,
            duration_per_frame=1.5
        )
        
        if success_ccw:
            file_size_ccw = Path(output_path_ccw).stat().st_size
            print(f"✅ Counterclockwise rotation video created!")
            print(f"   Output: {output_path_ccw}")
            print(f"   File size: {file_size_ccw:,} bytes ({file_size_ccw/1024:.1f} KB)")
            
            print("\n🎉 Both rotation tests completed successfully!")
            print(f"📁 Check the 'output' folder for:")
            print(f"   - {test_image_path} (test image)")
            print(f"   - {output_path} (clockwise rotation)")
            print(f"   - {output_path_ccw} (counterclockwise rotation)")
            
        else:
            print("❌ Failed to create counterclockwise rotation video")
            return False
            
    else:
        print("❌ Failed to create clockwise rotation video")
        return False
    
    return True


def show_usage_examples():
    """Show usage examples for the rotation feature."""
    print("\n📚 Usage Examples:")
    print("=" * 50)
    
    examples = [
        "# Basic clockwise rotation",
        "python telegram_animator.py image.jpg rotating_sticker.webm --rotation clockwise",
        "",
        "# Counterclockwise with custom duration", 
        "python telegram_animator.py image.jpg rotating_sticker.webm --rotation counterclockwise --rotation-duration 3.0",
        "",
        "# Smooth rotation with more steps",
        "python telegram_animator.py image.jpg rotating_sticker.webm --rotation clockwise --rotation-steps 48",
        "",
        "# Combined with background removal",
        "python telegram_animator.py person.jpg rotating_person.webm --rotation clockwise --remove-bg",
        "",
        "# High quality rotation",
        "python telegram_animator.py logo.png rotating_logo.webm --rotation clockwise --quality 9"
    ]
    
    for example in examples:
        print(f"  {example}")


if __name__ == "__main__":
    try:
        # Run the test
        success = test_rotation_animation()
        
        if success:
            show_usage_examples()
            print("\n🎯 Rotation feature is ready to use!")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
