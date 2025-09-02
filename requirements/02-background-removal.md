# Background Removal Module

## Implementation Goal
Provide optional AI-powered background removal using rembg library, with graceful fallback when disabled or unavailable.

## Technical Details

### Rembg Integration
```python
def remove_background(image: PIL.Image, model='u2net') -> PIL.Image:
    """
    - Use rembg library for AI background removal
    - Preserve alpha channel from removal process
    - Support different models (u2net, u2netp, u2net_human_seg)
    """
```

### Configuration Flag
```python
class BackgroundRemover:
    def __init__(self, enabled=True, model='u2net'):
        self.enabled = enabled
        self.model = model
        self._initialize_rembg()
    
    def _initialize_rembg(self):
        """
        - Try importing rembg
        - If ImportError, set enabled=False
        - Log warning but don't crash
        """
```

## Code Structure

```python
class BackgroundRemover:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.session = None
        
    def setup(self):
        """Initialize rembg session if enabled"""
        if self.enabled:
            try:
                from rembg import remove, new_session
                self.session = new_session('u2net')
                return True
            except ImportError:
                print("Warning: rembg not installed, disabling background removal")
                self.enabled = False
                return False
    
    def process(self, image: PIL.Image) -> PIL.Image:
        """Apply background removal if enabled"""
        if not self.enabled:
            return image
            
        try:
            from rembg import remove
            # Convert PIL to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Remove background
            output = remove(img_bytes.getvalue(), session=self.session)
            
            # Convert back to PIL
            return Image.open(io.BytesIO(output))
        except Exception as e:
            print(f"Background removal failed: {e}")
            return image
```

## Integration Points
- **Input**: PIL Image from image_processor
- **Output**: PIL Image with transparent background
- **Fallback**: Return original image if disabled/error

## Key Parameters

```python
REMBG_MODELS = {
    'u2net': 'General purpose (recommended)',
    'u2netp': 'Lightweight version',
    'u2net_human_seg': 'Optimized for people',
    'u2net_cloth_seg': 'Optimized for clothing'
}

DEFAULT_MODEL = 'u2net'
ALPHA_MATTING = True  # Improve edges
ALPHA_MATTING_FOREGROUND_THRESHOLD = 240
ALPHA_MATTING_BACKGROUND_THRESHOLD = 10
```

## Error Handling
- ImportError: rembg not installed → disable feature
- RuntimeError: Model download failed → use fallback
- MemoryError: Image too large → process in chunks or skip
- Always return valid image (original if processing fails)

## Implementation Example

```python
import io
from PIL import Image
from typing import Optional

class SmartBackgroundRemover:
    def __init__(self, enabled=True, model='u2net'):
        self.enabled = enabled
        self.model = model
        self.rembg_available = False
        self._check_availability()
    
    def _check_availability(self):
        if not self.enabled:
            return
        
        try:
            import rembg
            self.rembg_available = True
        except ImportError:
            self.rembg_available = False
            print("rembg not installed. Install with: pip install rembg")
    
    def remove_bg(self, image_path: str) -> Image.Image:
        img = Image.open(image_path)
        
        if not self.enabled or not self.rembg_available:
            # Convert to RGBA even if not removing background
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            return img
        
        from rembg import remove
        
        # Process with rembg
        with open(image_path, 'rb') as f:
            input_bytes = f.read()
        
        output_bytes = remove(input_bytes, model=self.model)
        return Image.open(io.BytesIO(output_bytes))
```

## CLI Integration
```python
# Command line flags
--remove-bg      # Enable background removal (default)
--no-remove-bg   # Disable background removal
--bg-model MODEL # Choose rembg model (u2net, u2netp, etc.)
