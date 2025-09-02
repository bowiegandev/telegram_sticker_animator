# Video Creation Module

## Implementation Goal
Generate WebM animations with VP9 codec from processed image frames, optimizing for Telegram's 256KB file size limit.

## Technical Details

### WebM/VP9 Setup
```python
def create_webm_video(frames: List[np.ndarray], fps=30, output_path='output.webm'):
    """
    - Use imageio-ffmpeg for VP9 encoding
    - Configure bitrate for size optimization
    - Enable transparency with yuva420p pixel format
    """
```

### Frame Sequencing
```python
class VideoCreator:
    def __init__(self, fps=30, duration_per_frame=0.1):
        self.fps = fps
        self.duration_per_frame = duration_per_frame
        
    def calculate_frame_count(self, num_images: int) -> int:
        """
        - Calculate total frames needed
        - frames_per_image = fps * duration_per_frame
        - total_frames = num_images * frames_per_image
        """
```

## Code Structure

```python
import cv2
import imageio
import numpy as np
from typing import List

class WebMCreator:
    def __init__(self, fps=30, quality=8):
        self.fps = fps
        self.quality = quality
        self.codec = 'libvpx-vp9'
        
    def create_animation(self, 
                        frames: List[np.ndarray], 
                        output_path: str,
                        duration_per_frame: float = 0.1) -> bool:
        """Main video creation pipeline"""
        # 1. Calculate frame repetitions
        # 2. Setup video writer with VP9 codec
        # 3. Write frames with proper timing
        # 4. Optimize file size
        # 5. Verify output < 256KB
    
    def optimize_quality(self, frames: List[np.ndarray], target_size: int) -> dict:
        """Adjust quality settings to meet file size"""
        # Start with high quality
        # Reduce if file too large
        # Return optimal settings
```

## VP9 Configuration

```python
VIDEO_CONFIG = {
    'codec': 'libvpx-vp9',
    'fps': 30,
    'pix_fmt': 'yuva420p',  # For transparency
    'crf': 31,  # Quality (0-63, lower=better)
    'b:v': '256k',  # Target bitrate
    'quality': 'good',
    'cpu-used': 1,
    'tile-columns': 2,
    'g': 128,  # GOP size
    'threads': 4
}

def get_ffmpeg_params(quality_level: int) -> dict:
    """Map quality level (1-10) to VP9 parameters"""
    crf_map = {
        10: 20,  # Highest quality
        8: 31,   # Default
        5: 40,   # Medium
        1: 50    # Lowest quality
    }
    return {
        '-vcodec': 'libvpx-vp9',
        '-crf': str(crf_map.get(quality_level, 31)),
        '-b:v': '0',  # Variable bitrate
        '-pix_fmt': 'yuva420p'
    }
```

## Integration Points
- **Input**: List of numpy arrays (RGBA) from image_processor
- **Output**: WebM file at specified path
- **Validation**: Check file size < 256KB

## Key Parameters

```python
# Telegram constraints
MAX_FILE_SIZE = 256 * 1024  # 256KB in bytes
MAX_DURATION = 3.0  # 3 seconds maximum
DIMENSIONS = (512, 512)

# Quality presets
QUALITY_PRESETS = {
    'high': {'crf': 20, 'cpu-used': 0},
    'medium': {'crf': 31, 'cpu-used': 1},
    'low': {'crf': 40, 'cpu-used': 2},
    'tiny': {'crf': 50, 'cpu-used': 4}
}

# Frame timing
DEFAULT_FPS = 30
DEFAULT_FRAME_DURATION = 0.1  # seconds
```

## File Size Optimization

```python
def optimize_for_size(frames: List[np.ndarray], 
                     max_size: int = 256000) -> dict:
    """
    Progressive quality reduction strategy:
    1. Try with quality=10 (highest)
    2. If file > max_size, reduce to quality=8
    3. Continue reducing until file fits
    4. If still too large at quality=1, reduce FPS
    5. Last resort: reduce number of frames
    """
    
    strategies = [
        {'quality': 10, 'fps': 30},
        {'quality': 8, 'fps': 30},
        {'quality': 5, 'fps': 30},
        {'quality': 3, 'fps': 25},
        {'quality': 2, 'fps': 20},
        {'quality': 1, 'fps': 15}
    ]
```

## Implementation Example

```python
import imageio
import numpy as np
from pathlib import Path

class TelegramWebMCreator:
    def __init__(self):
        self.writer = None
        
    def create_video(self, 
                    images: List[np.ndarray],
                    output_path: str,
                    fps: int = 30,
                    duration_per_frame: float = 0.1) -> bool:
        
        # Calculate frame repetitions
        frames_per_image = int(fps * duration_per_frame)
        
        # Setup writer with VP9 codec
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec='libvpx-vp9',
            quality=8,
            pixelformat='yuva420p',
            output_params=[
                '-auto-alt-ref', '0',
                '-crf', '31'
            ]
        )
        
        # Write frames
        for img in images:
            # Repeat frame for duration
            for _ in range(frames_per_image):
                writer.append_data(img)
        
        writer.close()
        
        # Check file size
        file_size = Path(output_path).stat().st_size
        if file_size > 256000:
            print(f"Warning: File size {file_size} exceeds 256KB limit")
            return False
        
        return True
```

## Error Handling
- FileNotFoundError: Invalid output path
- ValueError: Invalid codec or parameters
- RuntimeError: FFmpeg not installed or codec unavailable
- Automatic quality reduction if file too large
