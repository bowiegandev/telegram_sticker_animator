"""
Background Removal Module for Telegram Sticker Animator

Provides optional AI-powered background removal using rembg library,
with graceful fallback when disabled or unavailable.
"""

import io
import logging
from PIL import Image
from typing import Optional, Dict, Any
from pathlib import Path

from config import (
    REMBG_MODELS, 
    DEFAULT_REMBG_MODEL, 
    ALPHA_MATTING,
    ALPHA_MATTING_FOREGROUND_THRESHOLD,
    ALPHA_MATTING_BACKGROUND_THRESHOLD,
    ERROR_MESSAGES
)


class BackgroundRemover:
    """
    AI-powered background removal with graceful fallback support.
    
    Uses rembg library for background removal with configurable models
    and comprehensive error handling. Falls back gracefully when rembg
    is not available or encounters errors.
    """
    
    def __init__(self, enabled: bool = True, model: str = DEFAULT_REMBG_MODEL):
        """
        Initialize the BackgroundRemover.
        
        Args:
            enabled: Whether background removal is enabled
            model: rembg model to use (u2net, u2netp, etc.)
        """
        self.enabled = enabled
        self.model = model
        self.session = None
        self.rembg_available = False
        self.logger = logging.getLogger(__name__)
        
        # Initialize rembg if enabled
        if self.enabled:
            self._initialize_rembg()
    
    def _initialize_rembg(self) -> bool:
        """
        Initialize rembg session if enabled and available.
        
        Returns:
            True if rembg was successfully initialized, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Try importing rembg modules
            import rembg
            from rembg import remove, new_session
            
            # Validate model selection
            if self.model not in REMBG_MODELS:
                self.logger.warning(
                    f"Unknown rembg model '{self.model}'. Using default: {DEFAULT_REMBG_MODEL}"
                )
                self.model = DEFAULT_REMBG_MODEL
            
            # Create session for the specified model
            self.session = new_session(self.model)
            self.rembg_available = True
            
            self.logger.info(f"Background removal initialized with model: {self.model}")
            return True
            
        except ImportError as e:
            self.logger.warning(ERROR_MESSAGES['rembg_not_installed'])
            self.rembg_available = False
            self.enabled = False
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to initialize rembg: {e}")
            self.rembg_available = False
            self.enabled = False
            return False
    
    def setup(self) -> bool:
        """
        Initialize rembg session if enabled.
        
        Returns:
            True if setup was successful, False otherwise
        """
        return self._initialize_rembg()
    
    def process(self, image: Image.Image) -> Image.Image:
        """
        Apply background removal if enabled.
        
        Args:
            image: PIL Image to process
            
        Returns:
            PIL Image with background removed (if enabled) or original image
        """
        # Return original image if disabled or unavailable
        if not self.enabled or not self.rembg_available:
            return self._ensure_rgba(image)
        
        try:
            return self._remove_background(image)
            
        except MemoryError as e:
            self.logger.error(f"Memory error during background removal: {e}")
            self.logger.info("Trying to process image in reduced quality mode...")
            try:
                return self._remove_background_fallback(image)
            except Exception as fallback_e:
                self.logger.error(f"Fallback processing also failed: {fallback_e}")
                return self._ensure_rgba(image)
                
        except Exception as e:
            self.logger.error(f"Background removal failed: {e}")
            return self._ensure_rgba(image)
    
    def _remove_background(self, image: Image.Image) -> Image.Image:
        """
        Core background removal implementation.
        
        Args:
            image: PIL Image to process
            
        Returns:
            PIL Image with background removed
        """
        from rembg import remove
        
        # Convert PIL to bytes
        img_bytes = io.BytesIO()
        
        # Save in best quality format for processing
        if image.mode == 'RGBA':
            image.save(img_bytes, format='PNG')
        else:
            # Convert to RGB for JPEG compatibility if needed
            rgb_image = image.convert('RGB')
            rgb_image.save(img_bytes, format='JPEG', quality=95)
        
        img_bytes.seek(0)
        
        # Remove background using rembg
        if ALPHA_MATTING:
            # Use alpha matting for better edge quality
            output_bytes = remove(
                img_bytes.getvalue(), 
                session=self.session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=ALPHA_MATTING_FOREGROUND_THRESHOLD,
                alpha_matting_background_threshold=ALPHA_MATTING_BACKGROUND_THRESHOLD
            )
        else:
            # Standard background removal
            output_bytes = remove(img_bytes.getvalue(), session=self.session)
        
        # Convert back to PIL Image
        result_image = Image.open(io.BytesIO(output_bytes))
        
        # Ensure RGBA mode for transparency support
        if result_image.mode != 'RGBA':
            result_image = result_image.convert('RGBA')
        
        return result_image
    
    def _remove_background_fallback(self, image: Image.Image) -> Image.Image:
        """
        Fallback background removal for large images or memory constraints.
        
        Args:
            image: PIL Image to process
            
        Returns:
            PIL Image with background removed (reduced quality)
        """
        from rembg import remove
        
        # Reduce image size for processing
        original_size = image.size
        max_dimension = 1024
        
        if max(original_size) > max_dimension:
            # Calculate resize ratio
            ratio = max_dimension / max(original_size)
            new_size = tuple(int(dim * ratio) for dim in original_size)
            
            # Resize for processing
            small_image = image.resize(new_size, Image.LANCZOS)
            
            # Process smaller image
            img_bytes = io.BytesIO()
            small_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            output_bytes = remove(img_bytes.getvalue(), session=self.session)
            result_image = Image.open(io.BytesIO(output_bytes))
            
            # Resize back to original dimensions
            result_image = result_image.resize(original_size, Image.LANCZOS)
            
            return result_image.convert('RGBA')
        else:
            # If image is already small enough, use standard processing
            return self._remove_background(image)
    
    def _ensure_rgba(self, image: Image.Image) -> Image.Image:
        """
        Ensure image is in RGBA mode for transparency support.
        
        Args:
            image: PIL Image to convert
            
        Returns:
            PIL Image in RGBA mode
        """
        if image.mode != 'RGBA':
            return image.convert('RGBA')
        return image
    
    def remove_bg(self, image_path: str) -> Image.Image:
        """
        Remove background from image file.
        
        This method provides the interface specified in the requirements
        for standalone usage.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image with background removed or original if processing fails
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Process through main pipeline
            return self.process(img)
            
        except Exception as e:
            self.logger.error(f"Failed to process image {image_path}: {e}")
            # Try to return original image if possible
            try:
                img = Image.open(image_path)
                return self._ensure_rgba(img)
            except Exception as load_e:
                self.logger.error(f"Cannot load image {image_path}: {load_e}")
                raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            'enabled': self.enabled,
            'available': self.rembg_available,
            'model': self.model,
            'model_description': REMBG_MODELS.get(self.model, 'Unknown model'),
            'alpha_matting': ALPHA_MATTING,
            'supported_models': list(REMBG_MODELS.keys())
        }
    
    def change_model(self, new_model: str) -> bool:
        """
        Change the rembg model and reinitialize if necessary.
        
        Args:
            new_model: New model to use
            
        Returns:
            True if model change was successful
        """
        if new_model not in REMBG_MODELS:
            self.logger.error(f"Unsupported model: {new_model}")
            return False
        
        if new_model == self.model:
            return True  # No change needed
        
        self.model = new_model
        
        if self.enabled and self.rembg_available:
            # Reinitialize with new model
            return self._initialize_rembg()
        
        return True


class SmartBackgroundRemover(BackgroundRemover):
    """
    Enhanced background remover with additional smart features.
    
    This class provides the exact implementation example from the requirements
    while inheriting all the robust functionality from BackgroundRemover.
    """
    
    def __init__(self, enabled: bool = True, model: str = DEFAULT_REMBG_MODEL):
        """
        Initialize the SmartBackgroundRemover.
        
        Args:
            enabled: Whether background removal is enabled
            model: rembg model to use
        """
        super().__init__(enabled, model)
        self._check_availability()
    
    def _check_availability(self):
        """
        Check if rembg is available and print user-friendly messages.
        
        This method provides the exact interface shown in requirements.
        """
        if not self.enabled:
            return
        
        if not self.rembg_available:
            print("rembg not installed. Install with: pip install rembg")


def remove_background(image: Image.Image, model: str = 'u2net') -> Image.Image:
    """
    Standalone function for background removal.
    
    This provides the functional interface specified in the requirements.
    
    Args:
        image: PIL Image to process
        model: rembg model to use
        
    Returns:
        PIL Image with background removed
    """
    remover = BackgroundRemover(enabled=True, model=model)
    return remover.process(image)


# Example usage as shown in requirements
if __name__ == "__main__":
    # Test the background remover
    try:
        # Initialize with different models
        for model_name, description in REMBG_MODELS.items():
            print(f"\nTesting model: {model_name} - {description}")
            
            remover = BackgroundRemover(enabled=True, model=model_name)
            info = remover.get_model_info()
            
            print(f"  Enabled: {info['enabled']}")
            print(f"  Available: {info['available']}")
            print(f"  Model: {info['model']}")
            
            if info['available']:
                print(f"  ✅ {model_name} initialized successfully")
            else:
                print(f"  ⚠️  {model_name} not available (rembg not installed)")
        
        print(f"\n📋 Background removal module ready!")
        print(f"🔧 Configuration loaded from config.py")
        print(f"📦 Supports models: {list(REMBG_MODELS.keys())}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
