"""
Video Creation Module for Telegram Sticker Animator

Generates WebM animations with VP9 codec from processed image frames,
optimizing for Telegram's 256KB file size limit.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image

from config import (
    TELEGRAM_MAX_FILE_SIZE,
    TELEGRAM_MAX_DURATION,
    TELEGRAM_RECOMMENDED_FPS,
    VP9_QUALITY_LEVELS,
    ERROR_MESSAGES,
    get_config
)


class VideoCreator:
    """
    Base video creator with frame timing and sequencing functionality.
    """
    
    def __init__(self, fps: int = 30, duration_per_frame: float = 0.1):
        """
        Initialize the VideoCreator.
        
        Args:
            fps: Frames per second for video output
            duration_per_frame: How long each image should be displayed (seconds)
        """
        self.fps = fps
        self.duration_per_frame = duration_per_frame
        self.logger = logging.getLogger(__name__)
    
    def calculate_frame_count(self, num_images: int) -> int:
        """
        Calculate total frames needed for animation.
        
        Args:
            num_images: Number of input images
            
        Returns:
            Total number of video frames required
        """
        frames_per_image = int(self.fps * self.duration_per_frame)
        total_frames = num_images * frames_per_image
        return total_frames
    
    def get_frame_repetitions(self, num_images: int) -> List[int]:
        """
        Get the number of repetitions for each image frame.
        
        Args:
            num_images: Number of input images
            
        Returns:
            List with repetition count for each image
        """
        frames_per_image = int(self.fps * self.duration_per_frame)
        return [frames_per_image] * num_images
    
    def validate_duration(self, num_images: int) -> bool:
        """
        Validate that the total video duration doesn't exceed Telegram limits.
        
        Args:
            num_images: Number of input images
            
        Returns:
            True if duration is valid, False otherwise
        """
        total_duration = num_images * self.duration_per_frame
        if total_duration > TELEGRAM_MAX_DURATION:
            self.logger.warning(
                f"Video duration ({total_duration:.2f}s) exceeds Telegram limit "
                f"({TELEGRAM_MAX_DURATION}s)"
            )
            return False
        return True


class WebMCreator(VideoCreator):
    """
    WebM video creator with VP9 codec support.
    """
    
    def __init__(self, fps: int = 30, quality: int = 8):
        """
        Initialize the WebMCreator.
        
        Args:
            fps: Frames per second
            quality: Quality level (1-10, where 10 is highest)
        """
        super().__init__(fps)
        self.quality = quality
        self.codec = 'libvpx-vp9'
        self.writer = None
    
    def get_ffmpeg_params(self, quality_level: int) -> Dict[str, Any]:
        """
        Map quality level (1-10) to VP9 parameters.
        
        Args:
            quality_level: Quality level from 1 (lowest) to 10 (highest)
            
        Returns:
            Dictionary with FFmpeg parameters for VP9 encoding
        """
        vp9_settings = VP9_QUALITY_LEVELS.get(quality_level, VP9_QUALITY_LEVELS[8])
        
        return {
            'codec': 'libvpx-vp9',
            'crf': vp9_settings['crf'],
            'bitrate': vp9_settings['bitrate'],
            'pix_fmt': 'yuva420p',  # For transparency support
            'cpu-used': 1,          # Balance between speed and quality
            'tile-columns': 2,      # Optimize for parallel processing
            'g': 128,               # GOP size
            'threads': 4            # Use multiple threads
        }
    
    def create_animation(self, 
                        frames: List[np.ndarray], 
                        output_path: str,
                        duration_per_frame: float = 0.1) -> bool:
        """
        Main video creation pipeline.
        
        Args:
            frames: List of RGBA numpy arrays (H×W×4, uint8)
            output_path: Output file path for WebM video
            duration_per_frame: Duration to display each frame (seconds)
            
        Returns:
            True if video creation successful, False otherwise
        """
        if not frames:
            self.logger.error("No frames provided for video creation")
            return False
        
        self.duration_per_frame = duration_per_frame
        
        try:
            # 1. Validate inputs
            if not self._validate_frames(frames):
                return False
            
            # 2. Check duration limits
            if not self.validate_duration(len(frames)):
                return False
            
            # 3. Calculate frame repetitions
            frame_repetitions = self.get_frame_repetitions(len(frames))
            
            # 4. Setup video writer with VP9 codec
            success = self._setup_writer(output_path, self.quality)
            if not success:
                return False
            
            # 5. Write frames with proper timing
            success = self._write_frames(frames, frame_repetitions)
            if not success:
                return False
            
            # 6. Close writer
            self._close_writer()
            
            # 7. Verify output file exists
            if not Path(output_path).exists():
                self.logger.error(f"Output file was not created: {output_path}")
                return False
            
            self.logger.info(f"Video created successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create video: {e}")
            if self.writer:
                self._close_writer()
            return False
    
    def _validate_frames(self, frames: List[np.ndarray]) -> bool:
        """
        Validate input frames for video creation.
        
        Args:
            frames: List of numpy arrays to validate
            
        Returns:
            True if frames are valid, False otherwise
        """
        if not frames:
            self.logger.error("Empty frames list provided")
            return False
        
        expected_shape = frames[0].shape
        
        for i, frame in enumerate(frames):
            if frame.shape != expected_shape:
                self.logger.error(
                    f"Frame {i} has inconsistent shape: {frame.shape} "
                    f"(expected {expected_shape})"
                )
                return False
            
            if frame.dtype != np.uint8:
                self.logger.error(
                    f"Frame {i} has wrong dtype: {frame.dtype} (expected uint8)"
                )
                return False
            
            if len(frame.shape) != 3 or frame.shape[2] != 4:
                self.logger.error(
                    f"Frame {i} should be RGBA format (H×W×4), got {frame.shape}"
                )
                return False
        
        return True
    
    def _setup_writer(self, output_path: str, quality: int) -> bool:
        """
        Setup imageio writer with VP9 codec.
        
        Args:
            output_path: Output file path
            quality: Quality level (1-10)
            
        Returns:
            True if writer setup successful, False otherwise
        """
        try:
            import imageio
            
            # Get VP9 parameters
            ffmpeg_params = self.get_ffmpeg_params(quality)
            
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Format output parameters properly for imageio
            output_params = [
                '-crf', str(ffmpeg_params['crf']),
                '-b:v', ffmpeg_params['bitrate'],
                '-cpu-used', str(ffmpeg_params['cpu-used']),
                '-tile-columns', str(ffmpeg_params['tile-columns']),
                '-g', str(ffmpeg_params['g']),
                '-threads', str(ffmpeg_params['threads'])
            ]
            
            # Setup writer with proper parameters
            self.writer = imageio.get_writer(
                output_path,
                fps=self.fps,
                codec=ffmpeg_params['codec'],
                pixelformat=ffmpeg_params['pix_fmt'],
                output_params=output_params
            )
            
            return True
            
        except ImportError as e:
            self.logger.error("imageio not available for video creation")
            return False
        except Exception as e:
            self.logger.error(f"Failed to setup video writer: {e}")
            return False
    
    def _write_frames(self, frames: List[np.ndarray], repetitions: List[int]) -> bool:
        """
        Write frames to video with specified repetitions.
        
        Args:
            frames: List of RGBA numpy arrays
            repetitions: Number of times to repeat each frame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for i, (frame, repeat_count) in enumerate(zip(frames, repetitions)):
                # Write each frame the specified number of times
                for _ in range(repeat_count):
                    self.writer.append_data(frame)
                
                self.logger.debug(f"Wrote frame {i+1}/{len(frames)} ({repeat_count} times)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write frames to video: {e}")
            return False
    
    def _close_writer(self):
        """Close the video writer safely."""
        if self.writer:
            try:
                self.writer.close()
            except Exception as e:
                self.logger.error(f"Error closing video writer: {e}")
            finally:
                self.writer = None


class TelegramWebMCreator(WebMCreator):
    """
    WebM creator optimized for Telegram's specific requirements.
    
    Handles automatic file size optimization to meet 256KB limit.
    """
    
    MAX_FILE_SIZE = TELEGRAM_MAX_FILE_SIZE
    DIMENSIONS = (512, 512)
    
    # Quality presets for progressive optimization
    QUALITY_PRESETS = {
        'high': {'crf': 20, 'cpu-used': 0, 'quality': 10},
        'medium': {'crf': 31, 'cpu-used': 1, 'quality': 8},
        'low': {'crf': 40, 'cpu-used': 2, 'quality': 5},
        'tiny': {'crf': 50, 'cpu-used': 4, 'quality': 1}
    }
    
    def __init__(self):
        """Initialize TelegramWebMCreator with default settings."""
        super().__init__(fps=TELEGRAM_RECOMMENDED_FPS, quality=8)
    
    def create_video(self, 
                    images: List[np.ndarray],
                    output_path: str,
                    fps: int = 30,
                    duration_per_frame: float = 0.1) -> bool:
        """
        Create video with automatic size optimization for Telegram.
        
        Args:
            images: List of RGBA numpy arrays from ImageProcessor
            output_path: Output file path
            fps: Frames per second
            duration_per_frame: Duration per frame in seconds
            
        Returns:
            True if video creation successful and within size limit
        """
        self.fps = fps
        self.duration_per_frame = duration_per_frame
        
        # Try creating video with size optimization
        success = self.optimize_for_size(images, output_path)
        
        if success:
            file_size = Path(output_path).stat().st_size
            self.logger.info(
                f"Video created successfully: {output_path} "
                f"({file_size} bytes, {file_size/1024:.1f} KB)"
            )
            
            if file_size <= self.MAX_FILE_SIZE:
                return True
            else:
                self.logger.warning(
                    f"File size {file_size} bytes exceeds Telegram limit "
                    f"({self.MAX_FILE_SIZE} bytes)"
                )
                return False
        
        return False
    
    def optimize_for_size(self, frames: List[np.ndarray], 
                         output_path: str,
                         max_size: int = None) -> bool:
        """
        Progressive quality reduction strategy to meet file size limit.
        
        Args:
            frames: List of RGBA numpy arrays
            output_path: Output file path
            max_size: Maximum file size in bytes (default: Telegram limit)
            
        Returns:
            True if optimization successful, False otherwise
        """
        if max_size is None:
            max_size = self.MAX_FILE_SIZE
        
        # Progressive optimization strategies
        strategies = [
            {'quality': 10, 'fps': 30},
            {'quality': 8, 'fps': 30},
            {'quality': 5, 'fps': 30},
            {'quality': 3, 'fps': 25},
            {'quality': 2, 'fps': 20},
            {'quality': 1, 'fps': 15}
        ]
        
        for i, strategy in enumerate(strategies):
            self.logger.info(
                f"Trying optimization strategy {i+1}/{len(strategies)}: "
                f"quality={strategy['quality']}, fps={strategy['fps']}"
            )
            
            # Update settings
            self.fps = strategy['fps']
            self.quality = strategy['quality']
            
            # Create video with current settings
            success = self.create_animation(frames, output_path, self.duration_per_frame)
            
            if not success:
                self.logger.error(f"Failed to create video with strategy {i+1}")
                continue
            
            # Check file size
            if not Path(output_path).exists():
                continue
                
            file_size = Path(output_path).stat().st_size
            self.logger.info(f"Generated file size: {file_size} bytes ({file_size/1024:.1f} KB)")
            
            if file_size <= max_size:
                self.logger.info(f"File size optimization successful with strategy {i+1}")
                return True
            
            self.logger.info(f"File too large, trying next strategy...")
        
        # If all strategies failed, try reducing frame count as last resort
        self.logger.warning("All quality strategies failed, trying frame reduction...")
        return self._try_frame_reduction(frames, output_path, max_size)
    
    def _try_frame_reduction(self, frames: List[np.ndarray], 
                           output_path: str, max_size: int) -> bool:
        """
        Last resort: reduce number of frames to meet size limit.
        
        Args:
            frames: Original frames list
            output_path: Output file path
            max_size: Maximum file size
            
        Returns:
            True if successful with reduced frames
        """
        # Try reducing frames by half
        reduced_frames = frames[::2]  # Take every other frame
        
        if len(reduced_frames) < 2:
            self.logger.error("Cannot reduce frames further - too few frames")
            return False
        
        self.logger.info(f"Reducing frames from {len(frames)} to {len(reduced_frames)}")
        
        # Use lowest quality settings
        self.fps = 15
        self.quality = 1
        
        success = self.create_animation(reduced_frames, output_path, self.duration_per_frame)
        
        if success and Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            if file_size <= max_size:
                self.logger.info(f"Frame reduction successful: {file_size} bytes")
                return True
        
        self.logger.error("Failed to meet size limit even with frame reduction")
        return False


# Standalone functions for easy integration
def create_webm_video(frames: List[np.ndarray], 
                     fps: int = 30, 
                     output_path: str = 'output.webm') -> bool:
    """
    Create WebM video from image frames.
    
    Args:
        frames: List of RGBA numpy arrays
        fps: Frames per second
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    creator = TelegramWebMCreator()
    return creator.create_video(frames, output_path, fps=fps)


def create_telegram_video(images: List[np.ndarray],
                         output_path: str,
                         duration_per_frame: float = 0.1) -> bool:
    """
    Create Telegram-optimized WebM video.
    
    This is the main interface function for integration with ImageProcessor.
    
    Args:
        images: List of RGBA numpy arrays from ImageProcessor.process_batch()
        output_path: Output WebM file path
        duration_per_frame: How long to display each image (seconds)
        
    Returns:
        True if video creation successful and meets Telegram requirements
    """
    creator = TelegramWebMCreator()
    return creator.create_video(
        images, 
        output_path, 
        fps=TELEGRAM_RECOMMENDED_FPS,
        duration_per_frame=duration_per_frame
    )


# Example usage as shown in requirements
if __name__ == "__main__":
    # Test video creation
    print("🎬 Video Creation Module - Test")
    print("=" * 50)
    
    try:
        # Create test frames (simulating ImageProcessor output)
        test_frames = []
        for i in range(5):
            # Create a simple colored frame (512x512 RGBA)
            frame = np.zeros((512, 512, 4), dtype=np.uint8)
            frame[:, :, i % 3] = 255 * (i + 1) / 5  # Vary colors
            frame[:, :, 3] = 255  # Full opacity
            test_frames.append(frame)
        
        print(f"Created {len(test_frames)} test frames")
        
        # Test TelegramWebMCreator
        creator = TelegramWebMCreator()
        output_path = "test_output.webm"
        
        success = creator.create_video(
            test_frames, 
            output_path,
            duration_per_frame=0.2
        )
        
        if success:
            file_size = Path(output_path).stat().st_size
            print(f"✅ Test video created: {output_path}")
            print(f"📊 File size: {file_size} bytes ({file_size/1024:.1f} KB)")
            print(f"📏 Within Telegram limit: {file_size <= TELEGRAM_MAX_FILE_SIZE}")
        else:
            print("❌ Test video creation failed")
            
        # Clean up test file
        if Path(output_path).exists():
            Path(output_path).unlink()
            print("🧹 Test file cleaned up")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
