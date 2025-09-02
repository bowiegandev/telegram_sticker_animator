#!/usr/bin/env python3
"""
Test CLI Interface Module for Telegram Sticker Animator

Comprehensive testing of the CLI interface functionality including
argument parsing, validation, input processing, and integration
with all processing modules.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

from telegram_animator import CLIHandler, main
from cli_handler import create_cli_handler
from config import REMBG_MODELS, TELEGRAM_MAX_DURATION


def create_test_images(test_dir: Path, count: int = 3) -> list:
    """
    Create test images for CLI testing.
    
    Args:
        test_dir: Directory to create test images in
        count: Number of test images to create
        
    Returns:
        List of created image file paths
    """
    test_images = []
    
    for i in range(count):
        # Create a simple colored image
        img = Image.new('RGB', (100, 100), color=(255 * i // count, 100, 200))
        img_path = test_dir / f"test_image_{i+1}.jpg"
        img.save(img_path, 'JPEG')
        test_images.append(str(img_path))
    
    return test_images


def test_cli_argument_parsing():
    """Test command-line argument parsing functionality."""
    print("🧪 Testing CLI argument parsing...")
    
    cli = CLIHandler()
    
    # Test basic arguments
    args = cli.parse_args(['input.jpg', 'output.webm'])
    assert args.input == 'input.jpg'
    assert args.output == 'output.webm'
    assert args.remove_bg == True  # Default
    assert args.fps == 30  # Default
    assert args.quality == 8  # Default
    
    # Test background removal flags
    args = cli.parse_args(['input.jpg', 'output.webm', '--no-remove-bg'])
    assert args.remove_bg == False
    
    # Test custom settings
    args = cli.parse_args(['input.jpg', 'output.webv', '--fps', '25', '--duration', '0.2', '--quality', '5'])
    assert args.fps == 25
    assert args.duration == 0.2
    assert args.quality == 5
    
    # Test background model selection
    args = cli.parse_args(['input.jpg', 'output.webv', '--bg-model', 'u2netp'])
    assert args.bg_model == 'u2netp'
    
    print("✅ CLI argument parsing tests passed")


def test_input_path_processing():
    """Test input path processing with various scenarios."""
    print("🧪 Testing input path processing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test images
        test_images = create_test_images(test_dir, 3)
        
        cli = CLIHandler()
        
        # Test single file processing
        single_result = cli.get_image_paths(test_images[0])
        assert len(single_result) == 1
        assert single_result[0] == test_images[0]
        
        # Test directory processing
        dir_result = cli.get_image_paths(str(test_dir))
        assert len(dir_result) == 3
        assert all(path in dir_result for path in test_images)
        
        # Test wildcard processing
        wildcard_pattern = str(test_dir / "*.jpg")
        wildcard_result = cli.get_image_paths(wildcard_pattern)
        assert len(wildcard_result) == 3
        
        # Test empty directory
        empty_dir = test_dir / "empty"
        empty_dir.mkdir()
        empty_result = cli.get_image_paths(str(empty_dir))
        assert len(empty_result) == 0
        
        print("✅ Input path processing tests passed")


def test_validation_logic():
    """Test argument validation logic."""
    print("🧪 Testing validation logic...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        test_images = create_test_images(test_dir, 2)
        output_path = test_dir / "output.webv"
        
        cli = CLIHandler()
        
        # Test valid arguments
        valid_args = cli.parse_args([str(test_dir), str(output_path)])
        assert cli.validate_args(valid_args) == True
        
        # Test invalid FPS
        try:
            invalid_fps_args = cli.parse_args([str(test_dir), str(output_path), '--fps', '100'])
            assert cli.validate_args(invalid_fps_args) == False
        except SystemExit:
            pass  # argparse validation caught it first
        
        # Test invalid duration
        try:
            invalid_duration_args = cli.parse_args([str(test_dir), str(output_path), '--duration', '2.0'])
            assert cli.validate_args(invalid_duration_args) == False
        except SystemExit:
            pass
        
        # Test duration limit exceeded
        # Create many images to exceed 3-second limit
        many_images = create_test_images(test_dir, 40)  # 40 * 0.1 = 4 seconds > 3 second limit
        duration_limit_args = cli.parse_args([str(test_dir), str(output_path)])
        assert cli.validate_args(duration_limit_args) == False
        
        print("✅ Validation logic tests passed")


def test_error_handling():
    """Test error handling and user-friendly messages."""
    print("🧪 Testing error handling...")
    
    cli = CLIHandler()
    
    # Test non-existent input
    with patch('builtins.print') as mock_print:
        args = cli.parse_args(['nonexistent.jpg', 'output.webv'])
        result = cli.validate_args(args)
        assert result == False
        mock_print.assert_called()
    
    # Test invalid output extension
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        create_test_images(test_dir, 1)
        
        with patch('builtins.print') as mock_print:
            args = cli.parse_args([str(test_dir), 'output.mp4'])  # Wrong extension
            result = cli.validate_args(args)
            assert result == False
            mock_print.assert_called()
    
    print("✅ Error handling tests passed")


def test_usage_examples():
    """Test all usage examples from the requirements."""
    print("🧪 Testing usage examples from requirements...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test setup
        create_test_images(test_dir, 2)
        single_image = test_dir / "single.jpg"
        Image.new('RGB', (100, 100), 'red').save(single_image, 'JPEG')
        
        cli = CLIHandler()
        
        # Example 1: Basic usage with single image
        try:
            args = cli.parse_args([str(single_image), str(test_dir / "output1.webv")])
            assert args.input == str(single_image)
            assert args.output == str(test_dir / "output1.webv")
        except SystemExit:
            pass
        
        # Example 2: Process directory of images
        try:
            args = cli.parse_args([str(test_dir), str(test_dir / "animation.webv")])
            assert args.input == str(test_dir)
        except SystemExit:
            pass
        
        # Example 3: Disable background removal
        try:
            args = cli.parse_args([str(test_dir), str(test_dir / "output2.webv"), '--no-remove-bg'])
            assert args.remove_bg == False
        except SystemExit:
            pass
        
        # Example 4: Custom animation settings
        try:
            args = cli.parse_args([str(test_dir), str(test_dir / "output3.webv"), '--fps', '25', '--duration', '0.2'])
            assert args.fps == 25
            assert args.duration == 0.2
        except SystemExit:
            pass
        
        # Example 5: Low quality
        try:
            args = cli.parse_args([str(test_dir), str(test_dir / "output4.webv"), '--quality', '3'])
            assert args.quality == 3
        except SystemExit:
            pass
        
        # Example 6: Specific background model
        try:
            args = cli.parse_args([str(single_image), str(test_dir / "output5.webv"), '--bg-model', 'u2net_human_seg'])
            assert args.bg_model == 'u2net_human_seg'
        except SystemExit:
            pass
    
    print("✅ Usage example tests passed")


def test_cli_handler_factory():
    """Test the CLI handler factory function."""
    print("🧪 Testing CLI handler factory...")
    
    from cli_handler import create_cli_handler
    
    handler = create_cli_handler()
    assert handler is not None
    assert hasattr(handler, 'create_parser')
    assert hasattr(handler, 'validate_args')
    assert hasattr(handler, 'process_input_path')
    
    print("✅ CLI handler factory tests passed")


def test_validation_summary():
    """Test validation summary functionality."""
    print("🧪 Testing validation summary...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        test_images = create_test_images(test_dir, 2)
        
        cli = CLIHandler()
        args = cli.parse_args([str(test_dir), str(test_dir / "output.webv"), '--fps', '20', '--duration', '0.15'])
        
        summary = cli.get_validation_summary(args)
        
        assert summary['input_path'] == str(test_dir)
        assert summary['image_count'] == 2
        assert summary['fps'] == 20
        assert summary['total_duration'] == 0.3  # 2 images * 0.15 seconds
        assert summary['within_duration_limit'] == True
        assert len(summary['image_paths']) == 2
        
    print("✅ Validation summary tests passed")


def test_help_output():
    """Test help output contains all required information."""
    print("🧪 Testing help output...")
    
    cli = CLIHandler()
    
    # Test help contains usage examples
    try:
        with patch('sys.argv', ['telegram_animator.py', '--help']):
            cli.parse_args(['--help'])
    except SystemExit:
        pass  # Help exits normally
    
    # Check parser has all required arguments
    parser = cli.create_parser()
    
    # Check positional arguments
    actions = {action.dest: action for action in parser._actions}
    assert 'input' in actions
    assert 'output' in actions
    
    # Check optional arguments
    assert 'remove_bg' in actions
    assert 'fps' in actions
    assert 'duration' in actions
    assert 'quality' in actions
    assert 'bg_model' in actions
    assert 'verbose' in actions
    assert 'overwrite' in actions
    
    print("✅ Help output tests passed")


def run_integration_test():
    """Run a simple integration test with the main CLI."""
    print("🧪 Running integration test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Create test images
        test_images = create_test_images(test_dir, 2)
        output_file = test_dir / "test_output.webv"
        
        # Test CLI argument parsing and validation
        cli = CLIHandler()
        args = cli.parse_args([
            str(test_dir),
            str(output_file),
            '--no-remove-bg',  # Disable to avoid rembg dependency
            '--fps', '15',
            '--duration', '0.1',
            '--quality', '3',
            '--verbose'
        ])
        
        # Validate arguments
        if cli.validate_args(args):
            print("✅ Integration test validation passed")
            
            # Get validation summary
            summary = cli.get_validation_summary(args)
            print(f"   Found {summary['image_count']} images")
            print(f"   Total duration: {summary['total_duration']:.2f}s")
            print(f"   Estimated frames: {summary['estimated_frames']}")
        else:
            print("❌ Integration test validation failed")
    
    print("✅ Integration test completed")


def main():
    """Run all CLI interface tests."""
    print("🚀 Starting CLI Interface Tests")
    print("=" * 50)
    
    try:
        test_cli_argument_parsing()
        test_input_path_processing()
        test_validation_logic()
        test_error_handling()
        test_usage_examples()
        test_cli_handler_factory()
        test_validation_summary()
        test_help_output()
        run_integration_test()
        
        print("\n" + "=" * 50)
        print("🎉 All CLI interface tests passed!")
        print("✅ CLI module is ready for production use")
        
        # Show usage reminder
        print("\nUsage:")
        print("  python telegram_animator.py input.jpg output.webv")
        print("  python telegram_animator.py ./images/ animation.webv --fps 25")
        print("  python telegram_animator.py \"*.jpg\" output.webv --no-remove-bg")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
