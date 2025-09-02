# Telegram Sticker Animator - Project Overview

## Objective
Create a Python script that converts JPG/JPEG images into WebM animations suitable for Telegram animated stickers, with optional AI-powered background removal.

## Technical Specifications

### Telegram Sticker Requirements
- **Dimensions**: Exactly 512x512 pixels
- **Format**: WebM with VP9 codec
- **File Size**: Maximum 256KB
- **Frame Rate**: 30 FPS recommended
- **Duration**: 3 seconds maximum
- **Background**: Transparent (alpha channel support)

### Input/Output
- **Input**: JPG/JPEG image files (or PNG if already transparent)
- **Output**: Single WebM animation file
- **Processing**: Multiple images create animation sequence

## Architecture Overview

```
Input Images → Image Processor → Background Remover → Video Creator → WebM Output
                     ↓                    ↓               ↓
                  Resize/Pad      Optional Rembg      VP9 Encoding
```

## Core Dependencies

### Required
- **Pillow (PIL)**: Image loading, resizing, padding
- **opencv-python**: Video frame processing
- **imageio-ffmpeg**: WebM/VP9 encoding
- **numpy**: Array operations for frame data

### Optional
- **rembg**: AI-based background removal (can be disabled)

## Module Structure

```python
telegram_animator/
├── image_processor.py    # Core image operations
├── bg_remover.py         # Background removal logic
├── video_creator.py      # WebM generation
├── cli.py               # Command-line interface
└── config.py            # Default settings
```

## Configuration Defaults

```python
DEFAULT_CONFIG = {
    'output_size': (512, 512),
    'fps': 30,
    'frame_duration': 0.1,  # seconds per frame
    'quality': 8,            # 1-10 scale
    'remove_bg': True,       # can be disabled via CLI
    'codec': 'libvpx-vp9',
    'max_file_size': 256000  # 256KB in bytes
}
```

## Key Implementation Constraints

1. **Memory Efficiency**: Process images one at a time, not all at once
2. **Quality vs Size**: Balance quality settings to stay under 256KB
3. **Aspect Ratio**: Preserve original aspect when resizing
4. **Transparency**: Maintain alpha channel throughout pipeline
5. **Error Handling**: Graceful fallback if rembg unavailable
