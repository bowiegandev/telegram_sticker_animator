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

# Import advanced features
try:
    from interpolation_engine import interpolate_frame_sequence
    from transition_engine import apply_transitions_to_sequence
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False


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
    Advanced WebM creator optimized for Telegram's specific requirements.
    
    Handles automatic file size optimization and supports advanced features
    including frame interpolation, transitions, and preset animation modes.
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
    
    # Advanced animation presets
    ANIMATION_PRESETS = {
        'cinematic': {
            'fps': 24,
            'duration_per_frame': 0.4,
            'quality': 9,
            'interpolation': 'cubic',
            'interp_frames': 2,
            'transition': 'crossfade',
            'transition_duration': 0.5,
            'motion_blur': 0.3
        },
        'smooth': {
            'fps': 30,
            'duration_per_frame': 0.2,
            'quality': 8,
            'interpolation': 'cubic',
            'interp_frames': 3,
            'transition': 'crossfade',
            'transition_duration': 0.3,
            'motion_blur': 0.1
        },
        'slideshow': {
            'fps': 30,
            'duration_per_frame': 0.8,
            'quality': 9,
            'interpolation': 'linear',
            'interp_frames': 1,
            'transition': 'fade',
            'transition_duration': 0.4,
            'motion_blur': 0.0
        },
        'dynamic': {
            'fps': 25,
            'duration_per_frame': 0.3,
            'quality': 7,
            'interpolation': 'motion',
            'interp_frames': 2,
            'transition': 'scale',
            'transition_duration': 0.3,
            'motion_blur': 0.2
        }
    }
    
    def __init__(self, preset: Optional[str] = None):
        """
        Initialize TelegramWebMCreator with optional preset.
        
        Args:
            preset: Animation preset ('cinematic', 'smooth', 'slideshow', 'dynamic')
        """
        super().__init__(fps=TELEGRAM_RECOMMENDED_FPS, quality=8)
        
        # Advanced feature settings
        self.use_interpolation = False
        self.interpolation_type = 'linear'
        self.interpolation_frames = 2
        self.use_transitions = False
        self.transition_type = 'crossfade'
        self.transition_duration = 0.3
        self.transition_kwargs = {}
        self.motion_blur_intensity = 0.0
        
        # Rotation animation settings
        self.use_rotation = False
        self.rotation_direction = 'clockwise'
        self.rotation_duration = 2.0  # seconds for full 360° rotation
        self.rotation_steps = 36  # 10-degree increments for smooth rotation
        
        # Apply preset if specified
        if preset and preset in self.ANIMATION_PRESETS:
            self.apply_preset(preset)
    
    def apply_preset(self, preset_name: str):
        """
        Apply animation preset settings.
        
        Args:
            preset_name: Name of preset to apply
        """
        if preset_name not in self.ANIMATION_PRESETS:
            self.logger.warning(f"Unknown preset '{preset_name}', using default settings")
            return
        
        preset = self.ANIMATION_PRESETS[preset_name]
        
        # Basic settings
        self.fps = preset['fps']
        self.duration_per_frame = preset['duration_per_frame']
        self.quality = preset['quality']
        
        # Advanced features
        if ADVANCED_FEATURES_AVAILABLE:
            self.use_interpolation = preset.get('interp_frames', 0) > 0
            self.interpolation_type = preset.get('interpolation', 'linear')
            self.interpolation_frames = preset.get('interp_frames', 2)
            
            self.use_transitions = bool(preset.get('transition'))
            self.transition_type = preset.get('transition', 'crossfade')
            self.transition_duration = preset.get('transition_duration', 0.3)
            
            # Set transition-specific parameters
            if preset.get('transition') == 'scale':
                self.transition_kwargs = {'scale_type': 'zoom_in'}
            elif preset.get('transition') == 'slide':
                self.transition_kwargs = {'direction': 'left'}
            
            self.motion_blur_intensity = preset.get('motion_blur', 0.0)
            
            self.logger.info(f"Applied '{preset_name}' preset with advanced features")
        else:
            self.logger.info(f"Applied '{preset_name}' preset (basic features only)")
    
    def enable_interpolation(self, interpolation_type: str = 'cubic', num_frames: int = 2):
        """
        Enable frame interpolation.
        
        Args:
            interpolation_type: Type of interpolation ('linear', 'cubic', 'motion')
            num_frames: Number of intermediate frames to generate
        """
        if not ADVANCED_FEATURES_AVAILABLE:
            self.logger.warning("Advanced features not available, interpolation disabled")
            return
        
        self.use_interpolation = True
        self.interpolation_type = interpolation_type
        self.interpolation_frames = num_frames
        self.logger.info(f"Enabled {interpolation_type} interpolation with {num_frames} intermediate frames")
    
    def enable_transitions(self, transition_type: str = 'crossfade', duration: float = 0.3, **kwargs):
        """
        Enable transition effects.
        
        Args:
            transition_type: Type of transition ('fade', 'crossfade', 'slide', 'scale')
            duration: Duration of transitions in seconds
            **kwargs: Transition-specific parameters
        """
        if not ADVANCED_FEATURES_AVAILABLE:
            self.logger.warning("Advanced features not available, transitions disabled")
            return
        
        self.use_transitions = True
        self.transition_type = transition_type
        self.transition_duration = duration
        self.transition_kwargs = kwargs
        self.logger.info(f"Enabled {transition_type} transitions with {duration}s duration")
    
    def set_motion_blur(self, intensity: float):
        """
        Set motion blur intensity.
        
        Args:
            intensity: Motion blur intensity (0.0 to 1.0)
        """
        self.motion_blur_intensity = max(0.0, min(1.0, intensity))
        self.logger.info(f"Set motion blur intensity to {self.motion_blur_intensity}")
    
    def enable_rotation(self, direction: str = 'clockwise', duration: float = 2.0, steps: int = 36):
        """
        Enable 360-degree rotation animation.
        
        Args:
            direction: Rotation direction ('clockwise' or 'counterclockwise')
            duration: Duration for full 360-degree rotation in seconds
            steps: Number of rotation steps (higher = smoother rotation)
        """
        self.use_rotation = True
        self.rotation_direction = direction.lower()
        self.rotation_duration = duration
        self.rotation_steps = max(12, min(72, steps))  # Limit steps for reasonable file size
        
        if self.rotation_direction not in ['clockwise', 'counterclockwise']:
            self.logger.warning(f"Invalid rotation direction '{direction}', defaulting to 'clockwise'")
            self.rotation_direction = 'clockwise'
        
        self.logger.info(
            f"Enabled {self.rotation_direction} rotation: {duration}s duration, {self.rotation_steps} steps"
        )
    
    def _generate_rotation_frames(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Generate rotation frames from a single image.
        
        Args:
            image: Source image as RGBA numpy array (512x512x4)
            
        Returns:
            List of rotated frames for 360-degree animation
        """
        if not self.use_rotation:
            return [image]
        
        rotation_frames = []
        angle_step = 360.0 / self.rotation_steps
        
        # Convert numpy array to PIL Image for rotation
        pil_image = Image.fromarray(image, mode='RGBA')
        
        self.logger.info(f"Generating {self.rotation_steps} rotation frames...")
        
        for i in range(self.rotation_steps):
            # Calculate rotation angle
            if self.rotation_direction == 'clockwise':
                angle = i * angle_step
            else:  # counterclockwise
                angle = -i * angle_step
            
            # Rotate image with high-quality resampling
            # Use expand=False to maintain 512x512 dimensions
            # fillcolor=(0,0,0,0) for transparent background
            try:
                # Try modern PIL resampling constant
                rotated_pil = pil_image.rotate(
                    angle,
                    resample=Image.Resampling.LANCZOS,
                    expand=False,
                    fillcolor=(0, 0, 0, 0)
                )
            except AttributeError:
                # Fallback for older PIL versions
                rotated_pil = pil_image.rotate(
                    angle,
                    resample=Image.LANCZOS,
                    expand=False,
                    fillcolor=(0, 0, 0, 0)
                )
            except ValueError:
                # If LANCZOS not available, use BICUBIC as fallback
                rotated_pil = pil_image.rotate(
                    angle,
                    resample=Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC,
                    expand=False,
                    fillcolor=(0, 0, 0, 0)
                )
            
            # Convert back to numpy array
            rotated_frame = np.array(rotated_pil)
            rotation_frames.append(rotated_frame)
        
        self.logger.info(f"Generated {len(rotation_frames)} rotation frames")
        return rotation_frames
    
    def create_video(self, 
                    images: List[np.ndarray],
                    output_path: str,
                    fps: int = 30,
                    duration_per_frame: float = 0.1) -> bool:
        """
        Create advanced video with interpolation, transitions, and optimization.
        
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
        
        # Apply advanced features to frames
        processed_frames = self._apply_advanced_features(images)
        
        # Try creating video with size optimization
        success = self.optimize_for_size_advanced(processed_frames, output_path)
        
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
    
    def _apply_advanced_features(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply advanced features like interpolation and transitions to frames.
        
        Args:
            frames: Source frames
            
        Returns:
            Processed frames with advanced features applied
        """
        processed_frames = frames.copy()
        
        # Apply rotation animation (generate rotation frames from single image)
        if self.use_rotation:
            if len(frames) == 1:
                self.logger.info(f"Generating 360° rotation animation from single image...")
                rotation_frames = self._generate_rotation_frames(frames[0])
                processed_frames = rotation_frames
                # Update duration to match rotation duration
                self.duration_per_frame = self.rotation_duration / len(rotation_frames)
                self.logger.info(f"Rotation complete: 1 image -> {len(rotation_frames)} rotation frames")
            else:
                self.logger.warning("Rotation animation works best with single images. Using first image only.")
                rotation_frames = self._generate_rotation_frames(frames[0])
                processed_frames = rotation_frames
                self.duration_per_frame = self.rotation_duration / len(rotation_frames)
        
        # Apply frame interpolation
        if self.use_interpolation and ADVANCED_FEATURES_AVAILABLE:
            self.logger.info(f"Applying {self.interpolation_type} interpolation...")
            processed_frames = interpolate_frame_sequence(
                processed_frames,
                self.interpolation_type,
                self.interpolation_frames
            )
            self.logger.info(f"Interpolation complete: {len(frames)} -> {len(processed_frames)} frames")
        
        # Apply transitions
        if self.use_transitions and ADVANCED_FEATURES_AVAILABLE:
            self.logger.info(f"Applying {self.transition_type} transitions...")
            processed_frames = apply_transitions_to_sequence(
                processed_frames,
                self.transition_type,
                self.transition_duration,
                self.fps,
                **self.transition_kwargs
            )
            self.logger.info(f"Transitions complete: {len(processed_frames)} final frames")
        
        # Apply motion blur if enabled
        if self.motion_blur_intensity > 0:
            processed_frames = self._apply_motion_blur(processed_frames)
        
        return processed_frames
    
    def _apply_motion_blur(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply motion blur to frames for smoother animation.
        
        Args:
            frames: Input frames
            
        Returns:
            Frames with motion blur applied
        """
        if self.motion_blur_intensity == 0 or len(frames) < 2:
            return frames
        
        try:
            import cv2
            
            blurred_frames = []
            kernel_size = max(3, int(self.motion_blur_intensity * 15))
            
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            self.logger.info(f"Applying motion blur with kernel size {kernel_size}")
            
            for i, frame in enumerate(frames):
                if i == 0 or i == len(frames) - 1:
                    # Don't blur first and last frames
                    blurred_frames.append(frame)
                else:
                    # Apply horizontal motion blur
                    kernel = np.zeros((kernel_size, kernel_size))
                    kernel[kernel_size//2, :] = 1 / kernel_size
                    
                    # Apply to RGB channels only
                    rgb_frame = frame[:, :, :3]
                    alpha_frame = frame[:, :, 3]
                    
                    blurred_rgb = cv2.filter2D(rgb_frame, -1, kernel)
                    
                    # Reconstruct RGBA frame
                    blurred_frame = np.dstack([blurred_rgb, alpha_frame])
                    blurred_frames.append(blurred_frame.astype(np.uint8))
            
            return blurred_frames
            
        except ImportError:
            self.logger.warning("OpenCV not available, skipping motion blur")
            return frames
        except Exception as e:
            self.logger.error(f"Motion blur failed: {e}")
            return frames
    
    def optimize_for_size_advanced(self, frames: List[np.ndarray], 
                                 output_path: str,
                                 max_size: int = None) -> bool:
        """
        Advanced optimization considering the impact of advanced features.
        
        Args:
            frames: Processed frames with advanced features
            output_path: Output file path
            max_size: Maximum file size in bytes
            
        Returns:
            True if optimization successful
        """
        if max_size is None:
            max_size = self.MAX_FILE_SIZE
        
        # Advanced features may create many more frames, so use more aggressive strategies
        strategies = [
            {'quality': 8, 'fps': self.fps},
            {'quality': 6, 'fps': self.fps},
            {'quality': 5, 'fps': max(15, int(self.fps * 0.8))},
            {'quality': 4, 'fps': max(12, int(self.fps * 0.6))},
            {'quality': 3, 'fps': max(10, int(self.fps * 0.5))},
            {'quality': 2, 'fps': max(8, int(self.fps * 0.4))},
            {'quality': 1, 'fps': max(6, int(self.fps * 0.3))}
        ]
        
        original_use_interpolation = self.use_interpolation
        original_use_transitions = self.use_transitions
        
        for i, strategy in enumerate(strategies):
            self.logger.info(
                f"Advanced optimization strategy {i+1}/{len(strategies)}: "
                f"quality={strategy['quality']}, fps={strategy['fps']}"
            )
            
            # Update settings
            self.fps = strategy['fps']
            self.quality = strategy['quality']
            
            # For very aggressive strategies, disable advanced features
            if strategy['quality'] <= 3:
                current_frames = self._apply_basic_processing(frames)
            else:
                current_frames = frames
            
            # Create video with current settings
            success = self.create_animation(current_frames, output_path, self.duration_per_frame)
            
            if not success:
                self.logger.error(f"Failed with strategy {i+1}")
                continue
            
            # Check file size
            if not Path(output_path).exists():
                continue
            
            file_size = Path(output_path).stat().st_size
            self.logger.info(f"File size: {file_size} bytes ({file_size/1024:.1f} KB)")
            
            if file_size <= max_size:
                self.logger.info(f"Optimization successful with strategy {i+1}")
                return True
        
        # Restore original settings
        self.use_interpolation = original_use_interpolation
        self.use_transitions = original_use_transitions
        
        # Last resort: frame reduction
        return self._try_frame_reduction_advanced(frames, output_path, max_size)
    
    def _apply_basic_processing(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply only basic processing, skipping advanced features for size optimization.
        """
        # Skip interpolation and transitions for size optimization
        # Just apply basic frame repetition based on duration_per_frame
        return frames
    
    def _try_frame_reduction_advanced(self, frames: List[np.ndarray], 
                                    output_path: str, max_size: int) -> bool:
        """
        Advanced frame reduction strategy.
        """
        # Try different reduction ratios
        reduction_ratios = [2, 3, 4, 6, 8]
        
        for ratio in reduction_ratios:
            reduced_frames = frames[::ratio]
            
            if len(reduced_frames) < 2:
                continue
            
            self.logger.info(f"Trying frame reduction by {ratio}: {len(frames)} -> {len(reduced_frames)}")
            
            # Use minimal settings
            self.fps = max(8, int(self.fps / 2))
            self.quality = 1
            
            success = self.create_animation(reduced_frames, output_path, self.duration_per_frame)
            
            if success and Path(output_path).exists():
                file_size = Path(output_path).stat().st_size
                if file_size <= max_size:
                    self.logger.info(f"Frame reduction successful: {file_size} bytes")
                    return True
        
        self.logger.error("All frame reduction strategies failed")
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
        
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_output.webm"
        
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
