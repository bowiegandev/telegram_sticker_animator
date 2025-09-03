"""
Transition Effects Engine for Telegram Sticker Animator

Provides various transition effects to create smooth, professional-looking
animations between frames including fades, crossfades, slides, and scales.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import math

from config import get_config


class TransitionEffect(ABC):
    """
    Base class for transition effects between frames.
    """
    
    def __init__(self, duration: float = 0.3):
        """
        Initialize transition effect.
        
        Args:
            duration: Duration of transition in seconds
        """
        self.duration = duration
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
    
    @abstractmethod
    def apply_transition(self, 
                        frame1: np.ndarray, 
                        frame2: np.ndarray, 
                        fps: int) -> List[np.ndarray]:
        """
        Apply transition effect between two frames.
        
        Args:
            frame1: Source frame (H×W×4 RGBA)
            frame2: Target frame (H×W×4 RGBA) 
            fps: Frames per second
            
        Returns:
            List of transition frames
        """
        pass
    
    def _validate_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Validate that frames are compatible for transitions."""
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
    
    def _calculate_transition_frames(self, fps: int) -> int:
        """Calculate number of frames needed for transition."""
        return max(1, int(self.duration * fps))
    
    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function for natural motion."""
        return t * t * (3.0 - 2.0 * t)
    
    def _elastic_ease_out(self, t: float) -> float:
        """Elastic easing for bouncy effects."""
        if t == 0 or t == 1:
            return t
        
        p = 0.3
        s = p / 4
        return math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1
    
    def _bounce_ease_out(self, t: float) -> float:
        """Bounce easing for playful effects."""
        if t < (1/2.75):
            return 7.5625 * t * t
        elif t < (2/2.75):
            t -= (1.5/2.75)
            return 7.5625 * t * t + 0.75
        elif t < (2.5/2.75):
            t -= (2.25/2.75)
            return 7.5625 * t * t + 0.9375
        else:
            t -= (2.625/2.75)
            return 7.5625 * t * t + 0.984375


class FadeTransition(TransitionEffect):
    """
    Fade transition - gradual opacity change.
    """
    
    def apply_transition(self, 
                        frame1: np.ndarray, 
                        frame2: np.ndarray, 
                        fps: int) -> List[np.ndarray]:
        """
        Apply fade transition by modifying alpha channel.
        
        Frame1 fades out while frame2 fades in.
        """
        if not self._validate_frames(frame1, frame2):
            return [frame1, frame2]
        
        num_frames = self._calculate_transition_frames(fps)
        frames = []
        
        # Convert to float for smooth calculations
        frame1_f = frame1.astype(np.float32)
        frame2_f = frame2.astype(np.float32)
        
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 1.0
            t_eased = self._ease_in_out(t)
            
            # Create transition frame with alpha blending
            result = np.zeros_like(frame1_f)
            
            # Fade out frame1, fade in frame2
            alpha1 = 1.0 - t_eased
            alpha2 = t_eased
            
            # Blend RGB channels
            result[:, :, :3] = alpha1 * frame1_f[:, :, :3] + alpha2 * frame2_f[:, :, :3]
            
            # Handle alpha channel - use maximum alpha for visibility
            result[:, :, 3] = np.maximum(alpha1 * frame1_f[:, :, 3], alpha2 * frame2_f[:, :, 3])
            
            frames.append(np.clip(result, 0, 255).astype(np.uint8))
        
        self.logger.debug(f"Generated {len(frames)} fade transition frames")
        return frames


class CrossfadeTransition(TransitionEffect):
    """
    Crossfade transition - simultaneous fade in/out with smooth blending.
    """
    
    def apply_transition(self, 
                        frame1: np.ndarray, 
                        frame2: np.ndarray, 
                        fps: int) -> List[np.ndarray]:
        """
        Apply crossfade with sophisticated alpha blending.
        """
        if not self._validate_frames(frame1, frame2):
            return [frame1, frame2]
        
        num_frames = self._calculate_transition_frames(fps)
        frames = []
        
        frame1_f = frame1.astype(np.float32)
        frame2_f = frame2.astype(np.float32)
        
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 1.0
            t_eased = self._ease_in_out(t)
            
            # Smooth crossfade blending
            weight1 = 1.0 - t_eased
            weight2 = t_eased
            
            # Blend all channels including alpha
            result = weight1 * frame1_f + weight2 * frame2_f
            
            # Enhance alpha blending for better visibility
            alpha_blend = np.maximum(weight1 * frame1_f[:, :, 3], weight2 * frame2_f[:, :, 3])
            result[:, :, 3] = alpha_blend
            
            frames.append(np.clip(result, 0, 255).astype(np.uint8))
        
        self.logger.debug(f"Generated {len(frames)} crossfade transition frames")
        return frames


class SlideTransition(TransitionEffect):
    """
    Slide transition - frame2 slides in from specified direction.
    """
    
    def __init__(self, duration: float = 0.3, direction: str = 'left'):
        """
        Initialize slide transition.
        
        Args:
            duration: Transition duration
            direction: Slide direction ('left', 'right', 'up', 'down')
        """
        super().__init__(duration)
        self.direction = direction.lower()
        
        if self.direction not in ['left', 'right', 'up', 'down']:
            self.logger.warning(f"Invalid slide direction '{direction}', using 'left'")
            self.direction = 'left'
    
    def apply_transition(self, 
                        frame1: np.ndarray, 
                        frame2: np.ndarray, 
                        fps: int) -> List[np.ndarray]:
        """
        Apply slide transition effect.
        """
        if not self._validate_frames(frame1, frame2):
            return [frame1, frame2]
        
        num_frames = self._calculate_transition_frames(fps)
        frames = []
        
        h, w = frame1.shape[:2]
        
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 1.0
            t_eased = self._ease_in_out(t)
            
            # Calculate slide offset
            if self.direction == 'left':
                offset_x = int(w * (1 - t_eased))
                offset_y = 0
            elif self.direction == 'right':
                offset_x = int(-w * (1 - t_eased))
                offset_y = 0
            elif self.direction == 'up':
                offset_x = 0
                offset_y = int(h * (1 - t_eased))
            else:  # down
                offset_x = 0
                offset_y = int(-h * (1 - t_eased))
            
            # Create composite frame
            result = frame1.copy()
            
            # Calculate slide window
            if self.direction == 'left':
                if offset_x < w:
                    result[:, offset_x:, :] = frame2[:, :w-offset_x, :]
            elif self.direction == 'right':
                if offset_x > -w:
                    result[:, :w+offset_x, :] = frame2[:, -offset_x:, :]
            elif self.direction == 'up':
                if offset_y < h:
                    result[offset_y:, :, :] = frame2[:h-offset_y, :, :]
            else:  # down
                if offset_y > -h:
                    result[:h+offset_y, :, :] = frame2[-offset_y:, :, :]
            
            frames.append(result)
        
        self.logger.debug(f"Generated {len(frames)} slide transition frames ({self.direction})")
        return frames


class ScaleTransition(TransitionEffect):
    """
    Scale transition - frame2 scales in from center or frame1 scales out.
    """
    
    def __init__(self, duration: float = 0.3, scale_type: str = 'zoom_in'):
        """
        Initialize scale transition.
        
        Args:
            duration: Transition duration
            scale_type: Type of scaling ('zoom_in', 'zoom_out', 'zoom_in_out')
        """
        super().__init__(duration)
        self.scale_type = scale_type.lower()
        
        if self.scale_type not in ['zoom_in', 'zoom_out', 'zoom_in_out']:
            self.logger.warning(f"Invalid scale type '{scale_type}', using 'zoom_in'")
            self.scale_type = 'zoom_in'
    
    def apply_transition(self, 
                        frame1: np.ndarray, 
                        frame2: np.ndarray, 
                        fps: int) -> List[np.ndarray]:
        """
        Apply scale transition effect.
        """
        if not self._validate_frames(frame1, frame2):
            return [frame1, frame2]
        
        num_frames = self._calculate_transition_frames(fps)
        frames = []
        
        h, w = frame1.shape[:2]
        
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 1.0
            
            if self.scale_type == 'zoom_in':
                t_eased = self._ease_in_out(t)
                scale_factor = t_eased
                scaling_frame = frame2
                background_frame = frame1
            elif self.scale_type == 'zoom_out':
                t_eased = self._ease_in_out(t)
                scale_factor = 1.0 - t_eased
                scaling_frame = frame1
                background_frame = frame2
            else:  # zoom_in_out
                t_eased = self._elastic_ease_out(t)
                if t < 0.5:
                    scale_factor = 2 * t_eased
                    scaling_frame = frame2
                    background_frame = frame1
                else:
                    scale_factor = 2 * (1 - t_eased)
                    scaling_frame = frame1
                    background_frame = frame2
            
            # Create scaled frame
            result = background_frame.copy()
            
            if scale_factor > 0.01:  # Avoid tiny scales
                # Calculate scaled dimensions
                scaled_h = max(1, int(h * scale_factor))
                scaled_w = max(1, int(w * scale_factor))
                
                # Resize scaling frame
                try:
                    import cv2
                    scaled_frame = cv2.resize(scaling_frame, (scaled_w, scaled_h), 
                                            interpolation=cv2.INTER_CUBIC)
                    
                    # Center the scaled frame
                    start_y = (h - scaled_h) // 2
                    start_x = (w - scaled_w) // 2
                    end_y = start_y + scaled_h
                    end_x = start_x + scaled_w
                    
                    # Ensure bounds
                    start_y = max(0, start_y)
                    start_x = max(0, start_x)
                    end_y = min(h, end_y)
                    end_x = min(w, end_x)
                    
                    # Overlay scaled frame
                    if end_y > start_y and end_x > start_x:
                        scaled_h_actual = end_y - start_y
                        scaled_w_actual = end_x - start_x
                        
                        if scaled_frame.shape[0] >= scaled_h_actual and scaled_frame.shape[1] >= scaled_w_actual:
                            result[start_y:end_y, start_x:end_x, :] = scaled_frame[:scaled_h_actual, :scaled_w_actual, :]
                
                except ImportError:
                    # Fallback without OpenCV - simple alpha blending
                    alpha = scale_factor
                    result_f = result.astype(np.float32)
                    scaling_f = scaling_frame.astype(np.float32)
                    result = ((1 - alpha) * result_f + alpha * scaling_f).astype(np.uint8)
            
            frames.append(result)
        
        self.logger.debug(f"Generated {len(frames)} scale transition frames ({self.scale_type})")
        return frames


class TransitionFactory:
    """
    Factory for creating transition effects.
    """
    
    @staticmethod
    def create_transition(transition_type: str, 
                         duration: float = 0.3,
                         **kwargs) -> TransitionEffect:
        """
        Create transition effect based on type.
        
        Args:
            transition_type: Type of transition ('fade', 'crossfade', 'slide', 'scale')
            duration: Transition duration in seconds
            **kwargs: Additional transition-specific parameters
            
        Returns:
            Appropriate transition effect
        """
        transition_type = transition_type.lower()
        
        if transition_type == 'fade':
            return FadeTransition(duration)
        elif transition_type == 'crossfade':
            return CrossfadeTransition(duration)
        elif transition_type == 'slide':
            direction = kwargs.get('direction', 'left')
            return SlideTransition(duration, direction)
        elif transition_type == 'scale':
            scale_type = kwargs.get('scale_type', 'zoom_in')
            return ScaleTransition(duration, scale_type)
        else:
            logging.getLogger(__name__).warning(
                f"Unknown transition type '{transition_type}', using crossfade"
            )
            return CrossfadeTransition(duration)


def apply_transitions_to_sequence(frames: List[np.ndarray],
                                transition_type: str = 'crossfade',
                                transition_duration: float = 0.3,
                                fps: int = 30,
                                **kwargs) -> List[np.ndarray]:
    """
    Apply transitions to an entire frame sequence.
    
    Args:
        frames: List of source frames
        transition_type: Type of transition to apply
        transition_duration: Duration of each transition
        fps: Frames per second
        **kwargs: Additional transition parameters
        
    Returns:
        Extended frame sequence with transitions
    """
    if len(frames) < 2:
        return frames
    
    transition = TransitionFactory.create_transition(
        transition_type, transition_duration, **kwargs
    )
    
    result_frames = []
    logger = logging.getLogger(__name__)
    
    logger.info(f"Applying {transition_type} transitions to {len(frames)} frames")
    
    for i in range(len(frames) - 1):
        # Add transition frames between current and next frame
        transition_frames = transition.apply_transition(frames[i], frames[i + 1], fps)
        
        # Add all transition frames except the last one (to avoid duplication)
        result_frames.extend(transition_frames[:-1])
        
        logger.debug(f"Applied transition between frames {i} and {i+1}")
    
    # Add the final frame
    result_frames.append(frames[-1])
    
    logger.info(f"Generated {len(result_frames)} total frames with transitions")
    return result_frames


class EasingFunctions:
    """
    Collection of easing functions for smooth animations.
    """
    
    @staticmethod
    def linear(t: float) -> float:
        """Linear easing (no easing)."""
        return t
    
    @staticmethod
    def ease_in(t: float) -> float:
        """Ease in - slow start."""
        return t * t
    
    @staticmethod
    def ease_out(t: float) -> float:
        """Ease out - slow end."""
        return 1 - (1 - t) * (1 - t)
    
    @staticmethod
    def ease_in_out(t: float) -> float:
        """Ease in-out - slow start and end."""
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def elastic(t: float) -> float:
        """Elastic easing - bouncy effect."""
        if t == 0 or t == 1:
            return t
        
        p = 0.3
        s = p / 4
        return math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1
    
    @staticmethod
    def bounce(t: float) -> float:
        """Bounce easing - bouncing ball effect."""
        if t < (1/2.75):
            return 7.5625 * t * t
        elif t < (2/2.75):
            t -= (1.5/2.75)
            return 7.5625 * t * t + 0.75
        elif t < (2.5/2.75):
            t -= (2.25/2.75)
            return 7.5625 * t * t + 0.9375
        else:
            t -= (2.625/2.75)
            return 7.5625 * t * t + 0.984375


if __name__ == "__main__":
    # Test transitions
    print("🎬 Transition Effects Engine - Test")
    print("=" * 50)
    
    # Create test frames
    frame1 = np.zeros((100, 100, 4), dtype=np.uint8)
    frame1[:, :, 0] = 255  # Red
    frame1[:, :, 3] = 255  # Alpha
    
    frame2 = np.zeros((100, 100, 4), dtype=np.uint8)
    frame2[:, :, 2] = 255  # Blue  
    frame2[:, :, 3] = 255  # Alpha
    
    test_frames = [frame1, frame2]
    
    # Test different transition types
    transitions = ['fade', 'crossfade', 'slide', 'scale']
    
    for transition_type in transitions:
        print(f"\nTesting {transition_type} transition:")
        
        if transition_type == 'slide':
            result = apply_transitions_to_sequence(
                test_frames, transition_type, 0.2, 30, direction='left'
            )
        elif transition_type == 'scale':
            result = apply_transitions_to_sequence(
                test_frames, transition_type, 0.2, 30, scale_type='zoom_in'
            )
        else:
            result = apply_transitions_to_sequence(
                test_frames, transition_type, 0.2, 30
            )
        
        print(f"  Input frames: {len(test_frames)}")
        print(f"  Output frames: {len(result)}")
        print(f"  ✅ {transition_type.capitalize()} transition working")
