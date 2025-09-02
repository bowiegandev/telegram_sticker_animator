# How to Create WebM Animations from JPG Images

A simple guide to using the Telegram Animator Script to convert your JPG images into WebM animations perfect for Telegram stickers.

## Quick Start

### 1. Install Dependencies
```bash
# Create and activate virtual environment (recommended)
python -m venv telegram_animator_env

# Activate virtual environment
# On Git Bash (Windows):
source telegram_animator_env/Scripts/activate
# On Linux/Mac:
# source telegram_animator_env/bin/activate
# On Windows Command Prompt:
# telegram_animator_env\Scripts\activate.bat

# Install required packages
pip install -r requirements.txt

# Optional: For background removal
pip install rembg
```

### 2. Your First Animation
```bash
# Convert a single image
python telegram_animator.py my_image.jpg my_animation.webm

# Convert all images in a folder
python telegram_animator.py ./my_images/ my_animation.webm
```

That's it! Your WebM animation is ready for Telegram.

## Common Usage Examples

### Single Image to Animation
```bash
python telegram_animator.py photo.jpg sticker.webm
```
Creates a WebM animation from one image (useful for static stickers with transparency).

### Multiple Images to Animation
```bash
python telegram_animator.py ./image_sequence/ animated_sticker.webm
```
Converts all JPG/PNG images in the folder to an animated sequence.

### Using Wildcards
```bash
python telegram_animator.py "frame_*.jpg" animation.webm
```
Process specific images matching a pattern.

## Customizing Your Animation

### Control Animation Speed
```bash
# Slower animation (0.2 seconds per image)
python telegram_animator.py images/ slow_animation.webm --duration 0.2

# Faster animation (0.05 seconds per image) 
python telegram_animator.py images/ fast_animation.webm --duration 0.05

# Custom frame rate
python telegram_animator.py images/ smooth_animation.webm --fps 60
```

### Adjust Quality and File Size
```bash
# High quality (larger file)
python telegram_animator.py images/ hq_animation.webm --quality 9

# Low quality (smaller file, good for file size limits)
python telegram_animator.py images/ small_animation.webm --quality 3

# Medium quality (default, balanced)
python telegram_animator.py images/ animation.webm --quality 8
```

### Background Removal Options
```bash
# Automatic background removal (default)
python telegram_animator.py person.jpg clean_sticker.webm

# Disable background removal
python telegram_animator.py image.jpg animation.webm --no-remove-bg

# Use specific AI model for people
python telegram_animator.py portrait.jpg person_sticker.webm --bg-model u2net_human_seg
```

## Telegram Sticker Requirements

Your animations will automatically be optimized for Telegram stickers:
- **Size**: 512×512 pixels (auto-resized with padding)
- **File size**: Under 256KB (auto-compressed)
- **Duration**: Maximum 3 seconds
- **Format**: WebM with transparency support

## Tips for Best Results

### Preparing Your Images
1. **Use high-quality JPG/PNG images** - the script will resize them properly
2. **Name files alphabetically** if you want specific order (e.g., `frame_01.jpg`, `frame_02.jpg`)
3. **Keep sequences short** - remember the 3-second limit

### Getting Smaller File Sizes
```bash
# Try lower quality first
python telegram_animator.py images/ small.webm --quality 2

# Reduce frame rate
python telegram_animator.py images/ small.webm --fps 15 --quality 5

# Shorter duration per frame
python telegram_animator.py images/ small.webm --duration 0.08
```

### Better Animations
```bash
# Smooth motion
python telegram_animator.py images/ smooth.webm --fps 30 --duration 0.1

# Dramatic pause effect
python telegram_animator.py images/ dramatic.webm --duration 0.3
```

## Troubleshooting

### "File size too large"
- Lower quality: `--quality 3`
- Reduce FPS: `--fps 15` 
- Shorter duration: `--duration 0.05`

### "Duration exceeds 3 seconds"
- Use fewer images
- Reduce duration per frame: `--duration 0.08`
- The script automatically calculates: `total_duration = number_of_images × duration_per_image`

### "No images found"
- Check file extensions (only .jpg, .jpeg, .png supported)
- Verify folder path is correct
- Use quotes around wildcard patterns: `"*.jpg"`

### Background removal not working
- Install rembg: `pip install rembg`
- Or disable it: `--no-remove-bg`

### FFmpeg errors
- Install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
- Make sure it's in your system PATH

## Advanced Options

```bash
# See all options
python telegram_animator.py --help

# Verbose output for debugging
python telegram_animator.py images/ output.webm --verbose

# Force overwrite existing files
python telegram_animator.py images/ output.webm --overwrite
```

## Real Examples

### Create a waving hand sticker
```bash
# Folder with: wave_01.jpg, wave_02.jpg, wave_03.jpg
python telegram_animator.py ./wave_sequence/ wave_sticker.webm --duration 0.15
```

### Make a spinning logo
```bash
python telegram_animator.py "logo_frame_*.jpg" spinning_logo.webm --duration 0.1 --quality 7
```

### Convert a single photo with background removal
```bash
python telegram_animator.py selfie.jpg clean_avatar.webm --bg-model u2net_human_seg
```

---

**That's it!** You now have everything you need to create WebM animations from your JPG images. The script handles all the technical details automatically while giving you control over timing, quality, and effects.
