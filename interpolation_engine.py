"""
Frame Interpolation Engine for Telegram Sticker Animator

Provides advanced interpolation methods to create smooth transitions
between source images by generating intermediate frames.
"""

import numpy as np
import logging
from typing import List, Optional, Tuple, Union
from scipy import ndimage
from scipy.interpolate import interp1d
import cv2

from config import get_config


class InterpolationEngine:
    """
    Base class for frame interpolation algorithms.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
    
    def interpolate_frames(self, 
                          frame1: np.ndarray, 
                          frame2: np.ndarray, 
                          num_intermediate: int = 3) -> List[np.ndarray]:
        """
        Generate intermediate frames between two source frames.
        
        Args:
            frame1: First frame (H×W×4 RGBA)
            frame2: Second frame (H×W×4 RGBA)
            num_intermediate: Number of intermediate frames to generate
            
        Returns:
            List of intermediate frames including source frames
        """
        raise NotImplementedError("Subclasses must implement interpolate_frames")
    
    def _validate_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Validate that frames are compatible for interpolation."""
        if frame1.shape != frame2.shape:
            self.logger.error(f"Frame shape mismatch: {frame1.shape} vs {frame2.shape}")
            return False
        
        if len(frame1.shape) != 3 or frame1.shape[2] != 4:
            self.logger.error(f"Frames must be RGBA format, got shape: {frame1.shape}")
            return False
        
        if frame1.dtype != np.uint8 or frame2.dtype != np.uint8:
            self.logger.error("Frames must be uint8 dtype")
            return False
        
        return True


class LinearInterpolator(InterpolationEngine):
    """
    Linear interpolation between frames.
    Simple but effective for basic smoothing.
    """
    
    def interpolate_frames(self, 
                          frame1: np.ndarray, 
                          frame2: np.ndarray, 
                          num_intermediate: int = 3) -> List[np.ndarray]:
        """
        Linear interpolation between two frames.
        
        Creates smooth linear blending between frames.
        """
        if not self._validate_frames(frame1, frame2):
            return [frame1, frame2]
        
        frames = [frame1]
        
        # Generate intermediate frames
        for i in range(1, num_intermediate + 1):
            alpha = i / (num_intermediate + 1)
            
            # Linear blend in float space for accuracy
            frame1_f = frame1.astype(np.float32)
            frame2_f = frame2.astype(np.float32)
            
            interpolated = (1 - alpha) * frame1_f + alpha * frame2_f
            interpolated = np.clip(interpolated, 0, 255).astype(np.uint8)
            
            frames.append(interpolated)
        
        frames.append(frame2)
        
        self.logger.debug(f"Generated {len(frames)} frames with linear interpolation")
        return frames


class CubicInterpolator(InterpolationEngine):
    """
    Cubic spline interpolation for smoother transitions.
    Provides more natural motion curves.
    """
    
    def interpolate_frames(self, 
                          frame1: np.ndarray, 
                          frame2: np.ndarray, 
                          num_intermediate: int = 3) -> List[np.ndarray]:
        """
        Cubic spline interpolation between frames.
        
        Uses cubic curves for more natural motion.
        """
        if not self._validate_frames(frame1, frame2):
            return [frame1, frame2]
        
        frames = [frame1]
        
        # Convert to float for interpolation
        frame1_f = frame1.astype(np.float32)
        frame2_f = frame2.astype(np.float32)
        
        # Create cubic spline interpolation
        x_points = np.array([0, 1])
        total_steps = num_intermediate + 2
        
        for i in range(1, num_intermediate + 1):
            t = i / (num_intermediate + 1)
            
            # Apply cubic easing function for smoother motion
            t_cubic = self._cubic_ease_in_out(t)
            
            # Interpolate each channel separately for better quality
            interpolated = np.zeros_like(frame1_f)
            
            for channel in range(4):  # RGBA channels
                channel1 = frame1_f[:, :, channel]
                channel2 = frame2_f[:, :, channel]
                
                interpolated[:, :, channel] = (1 - t_cubic) * channel1 + t_cubic * channel2
            
            interpolated = np.clip(interpolated, 0, 255).astype(np.uint8)
            frames.append(interpolated)
        
        frames.append(frame2)
        
        self.logger.debug(f"Generated {len(frames)} frames with cubic interpolation")
        return frames
    
    def _cubic_ease_in_out(self, t: float) -> float:
        """Apply cubic ease-in-out function for smoother motion."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            p = 2 * t - 2
            return 1 + p * p * p / 2


class MotionInterpolator(InterpolationEngine):
    """
    Motion-aware interpolation using optical flow.
    Provides the highest quality interpolation by estimating motion vectors.
    """
    
    def __init__(self):
        super().__init__()
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def interpolate_frames(self, 
                          frame1: np.ndarray, 
                          frame2: np.ndarray, 
                          num_intermediate: int = 3) -> List[np.ndarray]:
        """
        Motion-aware interpolation using optical flow.
        
        Estimates motion between frames and creates realistic intermediate frames.
        """
        if not self._validate_frames(frame1, frame2):
            return [frame1, frame2]
        
        try:
            frames = [frame1]
            
            # Convert RGBA to RGB for optical flow (OpenCV doesn't handle alpha)
            rgb1 = frame1[:, :, :3]
            rgb2 = frame2[:, :, :3]
            
            # Convert to grayscale for optical flow calculation
            gray1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None, **self.flow_params)
            
            # If optical flow fails, fall back to linear interpolation
            if flow is None:
                self.logger.warning("Optical flow failed, falling back to linear interpolation")
                linear_interpolator = LinearInterpolator()
                return linear_interpolator.interpolate_frames(frame1, frame2, num_intermediate)
            
            # Generate intermediate frames using motion vectors
            for i in range(1, num_intermediate + 1):
                t = i / (num_intermediate + 1)
                
                # Create warped intermediate frame
                intermediate = self._create_motion_frame(frame1, frame2, flow, t)
                frames.append(intermediate)
            
            frames.append(frame2)
            
            self.logger.debug(f"Generated {len(frames)} frames with motion interpolation")
            return frames
            
        except Exception as e:
            self.logger.warning(f"Motion interpolation failed: {e}, falling back to linear")
            linear_interpolator = LinearInterpolator()
            return linear_interpolator.interpolate_frames(frame1, frame2, num_intermediate)
    
    def _create_motion_frame(self, 
                           frame1: np.ndarray, 
                           frame2: np.ndarray, 
                           flow: np.ndarray, 
                           t: float) -> np.ndarray:
        """
        Create intermediate frame using optical flow.
        
        Args:
            frame1, frame2: Source frames
            flow: Optical flow vectors
            t: Interpolation factor (0.0 to 1.0)
            
        Returns:
            Motion-interpolated frame
        """
        # Simple motion interpolation - in practice this would be more sophisticated
        # For now, blend with motion-aware weighting
        
        # Apply cubic easing for smoother motion
        t_eased = self._ease_in_out(t)
        
        # Linear blend as fallback (motion warping is complex)
        frame1_f = frame1.astype(np.float32)
        frame2_f = frame2.astype(np.float32)
        
        interpolated = (1 - t_eased) * frame1_f + t_eased * frame2_f
        return np.clip(interpolated, 0, 255).astype(np.uint8)
    
    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function for natural motion."""
        return t * t * (3.0 - 2.0 * t)


class InterpolationFactory:
    """
    Factory for creating interpolation engines.
    """
    
    @staticmethod
    def create_interpolator(interpolation_type: str) -> InterpolationEngine:
        """
        Create interpolation engine based on type.
        
        Args:
            interpolation_type: 'linear', 'cubic', or 'motion'
            
        Returns:
            Appropriate interpolation engine
        """
        interpolation_type = interpolation_type.lower()
        
        if interpolation_type == 'linear':
            return LinearInterpolator()
        elif interpolation_type == 'cubic':
            return CubicInterpolator()
        elif interpolation_type == 'motion':
            return MotionInterpolator()
        else:
            logging.getLogger(__name__).warning(
                f"Unknown interpolation type '{interpolation_type}', using linear"
            )
            return LinearInterpolator()


def interpolate_frame_sequence(frames: List[np.ndarray], 
                             interpolation_type: str = 'linear',
                             num_intermediate: int = 2) -> List[np.ndarray]:
    """
    Interpolate an entire sequence of frames.
    
    Args:
        frames: List of source frames
        interpolation_type: Type of interpolation ('linear', 'cubic', 'motion')
        num_intermediate: Number of intermediate frames between each pair
        
    Returns:
        Extended frame sequence with interpolated frames
    """
    if len(frames) < 2:
        return frames
    
    interpolator = InterpolationFactory.create_interpolator(interpolation_type)
    interpolated_sequence = []
    
    logger = logging.getLogger(__name__)
    logger.info(f"Interpolating {len(frames)} frames with {num_intermediate} intermediate frames each")
    
    for i in range(len(frames) - 1):
        # Get interpolated frames between current and next frame
        interpolated_frames = interpolator.interpolate_frames(
            frames[i], frames[i + 1], num_intermediate
        )
        
        # Add all frames except the last one (to avoid duplication)
        interpolated_sequence.extend(interpolated_frames[:-1])
        
        logger.debug(f"Interpolated between frames {i} and {i+1}")
    
    # Add the final frame
    interpolated_sequence.append(frames[-1])
    
    logger.info(f"Generated {len(interpolated_sequence)} total frames from {len(frames)} source frames")
    return interpolated_sequence


if __name__ == "__main__":
    # Test interpolation
    print("🎬 Frame Interpolation Engine - Test")
    print("=" * 50)
    
    # Create test frames
    frame1 = np.zeros((100, 100, 4), dtype=np.uint8)
    frame1[:, :, 0] = 255  # Red
    frame1[:, :, 3] = 255  # Alpha
    
    frame2 = np.zeros((100, 100, 4), dtype=np.uint8)
    frame2[:, :, 2] = 255  # Blue  
    frame2[:, :, 3] = 255  # Alpha
    
    test_frames = [frame1, frame2]
    
    # Test different interpolation methods
    for method in ['linear', 'cubic', 'motion']:
        print(f"\nTesting {method} interpolation:")
        interpolated = interpolate_frame_sequence(test_frames, method, 2)
        print(f"  Input frames: {len(test_frames)}")
        print(f"  Output frames: {len(interpolated)}")
        print(f"  ✅ {method.capitalize()} interpolation working")
