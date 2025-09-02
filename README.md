# Telegram Animator Script

Convert JPG/JPEG images to WebM animations for Telegram stickers with optional AI-powered background removal.

## Project Structure

```
telegram_animator_script/
├── requirements/              # Development requirements (AI-optimized)
│   ├── 00-overview.md        # Project specifications
│   ├── 01-core-image-processing.md
│   ├── 02-background-removal.md
│   ├── 03-video-creation.md
│   └── 04-cli-interface.md
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Features

- **Image Processing**: Resize images to 512x512 with aspect ratio preservation
- **Background Removal**: Optional AI-powered background removal using rembg
- **WebM Creation**: Generate VP9-encoded WebM files optimized for Telegram
- **File Size Optimization**: Automatic quality adjustment to stay under 256KB
- **Batch Processing**: Convert multiple images into animation sequences
- **CLI Interface**: User-friendly command-line interface with configuration options

## Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# For background removal (optional)
pip install rembg
```

## Implementation Guide

The project is divided into modular requirements for optimal AI-assisted development:

1. **Start with Core Image Processing** (`requirements/01-core-image-processing.md`)
   - Implement basic image loading and resizing
   - Add transparent padding functionality

2. **Add Background Removal** (`requirements/02-background-removal.md`)
   - Integrate optional rembg support
   - Implement graceful fallback

3. **Implement Video Creation** (`requirements/03-video-creation.md`)
   - Setup WebM/VP9 encoding
   - Add file size optimization

4. **Create CLI Interface** (`requirements/04-cli-interface.md`)
   - Parse command-line arguments
   - Wire up all modules

## Usage (After Implementation)

```bash
# Basic usage
python telegram_animator.py image.jpg output.webm

# Process directory of images
python telegram_animator.py ./images/ animation.webm

# Disable background removal
python telegram_animator.py images/ output.webm --no-remove-bg

# Custom settings
python telegram_animator.py images/ output.webm --fps 25 --duration 0.2 --quality 5
```

## Technical Specifications

- **Output Format**: WebM with VP9 codec
- **Dimensions**: 512x512 pixels
- **File Size**: Maximum 256KB
- **Frame Rate**: 30 FPS (configurable)
- **Duration**: 3 seconds maximum
- **Transparency**: Full alpha channel support

## Development Notes

Each requirement file in the `requirements/` directory contains:
- Implementation goals and technical details
- Code structure and examples
- Integration points between modules
- Key parameters and configurations
- Error handling strategies

These files are optimized for AI-assisted development, providing clear context and implementation details for each module.
