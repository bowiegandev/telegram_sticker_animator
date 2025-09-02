"""
Configuration file for Telegram Sticker Animator

Contains all default settings, constants, and configuration options
for the image processing and video creation pipeline.
"""

from typing import Tuple

# Image Processing Configuration
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png']
DEFAULT_SIZE = (512, 512)
RESAMPLING_FILTER = 'LANCZOS'  # PIL.Image.LANCZOS
TRANSPARENT_COLOR = (0, 0, 0, 0)  # RGBA transparent

# Telegram Sticker Requirements
TELEGRAM_STICKER_SIZE = (512, 512)
TELEGRAM_MAX_FILE_SIZE = 256000  # 256KB in bytes
TELEGRAM_MAX_DURATION = 3.0  # seconds
TELEGRAM_RECOMMENDED_FPS = 30

# Video Creation Configuration
DEFAULT_CONFIG = {
    'output_size': (512, 512),
    'fps': 30,
    'frame_duration': 0.1,  # seconds per frame
    'quality': 8,            # 1-10 scale for VP9
    'remove_bg': True,       # can be disabled via CLI
    'codec': 'libvpx-vp9',
    'max_file_size': 256000  # 256KB in bytes
}

# Background Removal Configuration
REMBG_MODELS = {
    'u2net': 'General purpose (recommended)',
    'u2netp': 'Lightweight version',
    'u2net_human_seg': 'Optimized for people',
    'u2net_cloth_seg': 'Optimized for clothing'
}

DEFAULT_REMBG_MODEL = 'u2net'
ALPHA_MATTING = True  # Improve edges
ALPHA_MATTING_FOREGROUND_THRESHOLD = 240
ALPHA_MATTING_BACKGROUND_THRESHOLD = 10

# Video Encoding Settings
VP9_QUALITY_LEVELS = {
    1: {'crf': 50, 'bitrate': '50k'},   # Lowest quality, smallest file
    2: {'crf': 45, 'bitrate': '75k'},
    3: {'crf': 40, 'bitrate': '100k'},
    4: {'crf': 35, 'bitrate': '150k'},
    5: {'crf': 30, 'bitrate': '200k'},  # Balanced
    6: {'crf': 25, 'bitrate': '300k'},
    7: {'crf': 20, 'bitrate': '400k'},
    8: {'crf': 15, 'bitrate': '500k'},  # Default
    9: {'crf': 10, 'bitrate': '750k'},
    10: {'crf': 5, 'bitrate': '1000k'}  # Highest quality
}

# CLI Configuration
DEFAULT_CLI_SETTINGS = {
    'verbose': False,
    'overwrite': False,
    'remove_bg': True,
    'bg_model': DEFAULT_REMBG_MODEL,
    'fps': TELEGRAM_RECOMMENDED_FPS,
    'frame_duration': 0.1,
    'quality': 8,
    'output_format': 'webm'
}

# Error Messages
ERROR_MESSAGES = {
    'file_not_found': "Image file not found: {}",
    'unsupported_format': "Unsupported image format: {}. Supported formats: {}",
    'corrupted_image': "Cannot identify image file: {}. File may be corrupted.",
    'memory_error': "Image too large to process: {}. Try reducing image size.",
    'rembg_not_installed': "rembg not installed. Install with: pip install rembg",
    'ffmpeg_not_found': "FFmpeg not found. Please install FFmpeg for video creation.",
    'file_too_large': "Output file size ({} KB) exceeds Telegram limit (256 KB)",
    'processing_failed': "Failed to process image: {}",
    'batch_processing_failed': "Batch processing failed for {} out of {} images"
}

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
LOG_FILE = 'telegram_animator.log'

# Performance Settings
MAX_IMAGE_DIMENSION = 4096  # Maximum input image dimension
MAX_BATCH_SIZE = 100        # Maximum number of images to process at once
MEMORY_LIMIT_MB = 512       # Approximate memory limit for processing

class Config:
    """
    Configuration class for easy access to settings.
    
    Provides both class-level constants and instance-level
    configuration that can be modified at runtime.
    """
    
    def __init__(self):
        """Initialize with default configuration values."""
        self.image_formats = IMAGE_FORMATS.copy()
        self.target_size = DEFAULT_SIZE
        self.transparent_color = TRANSPARENT_COLOR
        self.remove_bg_enabled = True
        self.rembg_model = DEFAULT_REMBG_MODEL
        self.fps = TELEGRAM_RECOMMENDED_FPS
        self.frame_duration = 0.1
        self.quality = 8
        self.max_file_size = TELEGRAM_MAX_FILE_SIZE
        self.verbose = False
    
    def update(self, **kwargs):
        """
        Update configuration values.
        
        Args:
            **kwargs: Configuration key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration option: {key}")
    
    def get_vp9_settings(self) -> dict:
        """
        Get VP9 encoding settings based on quality level.
        
        Returns:
            Dictionary with CRF and bitrate settings
        """
        return VP9_QUALITY_LEVELS.get(self.quality, VP9_QUALITY_LEVELS[8])
    
    def validate(self):
        """
        Validate current configuration settings.
        
        Raises:
            ValueError: If any configuration value is invalid
        """
        if not isinstance(self.target_size, tuple) or len(self.target_size) != 2:
            raise ValueError("target_size must be a tuple of (width, height)")
        
        if self.quality not in range(1, 11):
            raise ValueError("quality must be between 1 and 10")
        
        if self.fps <= 0 or self.fps > 60:
            raise ValueError("fps must be between 1 and 60")
        
        if self.frame_duration <= 0:
            raise ValueError("frame_duration must be positive")
        
        if self.rembg_model not in REMBG_MODELS:
            raise ValueError(f"rembg_model must be one of: {list(REMBG_MODELS.keys())}")


# Global configuration instance
default_config = Config()


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Config instance with current settings
    """
    return default_config


def update_config(**kwargs):
    """
    Update the global configuration.
    
    Args:
        **kwargs: Configuration key-value pairs to update
    """
    default_config.update(**kwargs)


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print(f"Default target size: {config.target_size}")
    print(f"Default VP9 settings: {config.get_vp9_settings()}")
    print(f"Supported formats: {config.image_formats}")
    
    # Test validation
    try:
        config.validate()
        print("Configuration validation passed!")
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
