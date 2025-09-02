# Core Image Processing Module

## Implementation Goal
Handle image loading, validation, resizing to 512x512 pixels while preserving aspect ratio, and applying transparent padding as needed.

## Technical Details

### Image Loading
```python
def load_image(filepath: str) -> PIL.Image:
    """
    - Support JPG, JPEG, PNG formats
    - Convert to RGBA mode for transparency support
    - Validate image readability
    """
```

### Resize Algorithm
```python
def resize_with_aspect_ratio(image: PIL.Image, target_size: tuple) -> PIL.Image:
    """
    1. Calculate aspect ratio
    2. Determine new dimensions (fit within 512x512)
    3. Use PIL.Image.LANCZOS for high-quality downsampling
    4. Center image with transparent padding
    """
```

### Padding Implementation
```python
def add_transparent_padding(image: PIL.Image, target_size: tuple) -> PIL.Image:
    """
    - Create new RGBA image with target dimensions
    - Fill with transparent pixels (0, 0, 0, 0)
    - Paste resized image at center position
    - Calculate offsets: (target_width - img_width) // 2
    """
```

## Code Structure

```python
class ImageProcessor:
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
    
    def process_image(self, image_path: str) -> np.ndarray:
        """Main processing pipeline"""
        # 1. Load image
        # 2. Convert to RGBA
        # 3. Resize maintaining aspect
        # 4. Add padding if needed
        # 5. Return as numpy array for video processing
    
    def process_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        """Process multiple images"""
```

## Integration Points
- **Input**: File paths from CLI or directory listing
- **Output**: Numpy arrays (RGBA) for video_creator module
- **Optional**: Pass to bg_remover before final processing

## Key Parameters

```python
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png']
DEFAULT_SIZE = (512, 512)
RESAMPLING_FILTER = PIL.Image.LANCZOS
TRANSPARENT_COLOR = (0, 0, 0, 0)
```

## Error Handling
- FileNotFoundError: Invalid image path
- PIL.UnidentifiedImageError: Corrupted/unsupported format
- MemoryError: Image too large
- Return None or raise with descriptive message

## Implementation Example

```python
from PIL import Image
import numpy as np
from pathlib import Path

def process_single_image(image_path: str) -> np.ndarray:
    # Load and convert
    img = Image.open(image_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Calculate resize dimensions
    width, height = img.size
    aspect = width / height
    
    if aspect > 1:  # Wider than tall
        new_width = 512
        new_height = int(512 / aspect)
    else:  # Taller than wide
        new_height = 512
        new_width = int(512 * aspect)
    
    # Resize
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create padded image
    final = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    offset_x = (512 - new_width) // 2
    offset_y = (512 - new_height) // 2
    final.paste(img, (offset_x, offset_y))
    
    return np.array(final)
