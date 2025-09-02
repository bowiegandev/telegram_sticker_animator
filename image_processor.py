"""
Core Image Processing Module for Telegram Sticker Animator

Handles image loading, validation, resizing to 512x512 pixels while preserving 
aspect ratio, and applying transparent padding as needed.
"""

import io
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from typing import List, Optional, Tuple, Union


class ImageProcessor:
    """
    Processes images for Telegram sticker animation creation.
    
    Handles loading, resizing with aspect ratio preservation, 
    and transparent padding to create 512x512 RGBA images.
    """
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}
    
    # Default processing settings
    DEFAULT_SIZE = (512, 512)
    RESAMPLING_FILTER = Image.LANCZOS
    TRANSPARENT_COLOR = (0, 0, 0, 0)  # RGBA transparent
    
    def __init__(self, target_size: Tuple[int, int] = DEFAULT_SIZE):
        """
        Initialize the ImageProcessor.
        
        Args:
            target_size: Target dimensions for output images (width, height)
        """
        self.target_size = target_size
    
    def load_image(self, filepath: str) -> Image.Image:
        """
        Load and validate an image file.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            PIL Image object in RGBA mode
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file format is not supported
            PIL.UnidentifiedImageError: If image is corrupted/unreadable
        """
        path = Path(filepath)
        
        # Check file existence
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {filepath}")
        
        # Check file format
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        try:
            # Load image
            img = Image.open(filepath)
            
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            return img
            
        except Image.UnidentifiedImageError as e:
            raise Image.UnidentifiedImageError(
                f"Cannot identify image file: {filepath}. "
                "File may be corrupted or in an unsupported format."
            ) from e
        except MemoryError as e:
            raise MemoryError(
                f"Image too large to process: {filepath}. "
                "Try reducing image size or available memory."
            ) from e
    
    def resize_with_aspect_ratio(
        self, 
        image: Image.Image, 
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Resize image maintaining aspect ratio to fit within target dimensions.
        
        Args:
            image: PIL Image to resize
            target_size: Target dimensions (width, height)
            
        Returns:
            Resized PIL Image (may be smaller than target_size)
        """
        target_width, target_height = target_size
        original_width, original_height = image.size
        
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        # Determine new dimensions to fit within target size
        if aspect_ratio > 1:  # Image is wider than tall
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:  # Image is taller than wide or square
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # Ensure dimensions don't exceed target
        if new_width > target_width:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        
        if new_height > target_height:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # Resize using high-quality LANCZOS filter
        return image.resize((new_width, new_height), self.RESAMPLING_FILTER)
    
    def add_transparent_padding(
        self, 
        image: Image.Image, 
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Add transparent padding to center image within target dimensions.
        
        Args:
            image: PIL Image to pad
            target_size: Target dimensions (width, height)
            
        Returns:
            PIL Image with transparent padding, exactly target_size dimensions
        """
        target_width, target_height = target_size
        img_width, img_height = image.size
        
        # Create new transparent image with target dimensions
        padded_image = Image.new('RGBA', target_size, self.TRANSPARENT_COLOR)
        
        # Calculate center position offsets
        offset_x = (target_width - img_width) // 2
        offset_y = (target_height - img_height) // 2
        
        # Paste the original image at center position
        padded_image.paste(image, (offset_x, offset_y), image)
        
        return padded_image
    
    def process_image(self, image_path: str) -> np.ndarray:
        """
        Main processing pipeline for a single image.
        
        Loads, converts to RGBA, resizes maintaining aspect ratio,
        adds padding, and returns as numpy array for video processing.
        
        Args:
            image_path: Path to the input image file
            
        Returns:
            Numpy array (RGBA) of processed image, shape (height, width, 4)
            
        Raises:
            Various exceptions from load_image method
        """
        # 1. Load image and convert to RGBA
        image = self.load_image(image_path)
        
        # 2. Resize maintaining aspect ratio
        resized_image = self.resize_with_aspect_ratio(image, self.target_size)
        
        # 3. Add transparent padding to reach exact target size
        final_image = self.add_transparent_padding(resized_image, self.target_size)
        
        # 4. Convert to numpy array for video processing
        return np.array(final_image)
    
    def process_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Process multiple images using the same pipeline.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of numpy arrays (RGBA) for each processed image
            
        Note:
            Failed images are skipped and not included in the output.
            Check logs for error messages about skipped images.
        """
        processed_images = []
        
        for i, image_path in enumerate(image_paths):
            try:
                processed_array = self.process_image(image_path)
                processed_images.append(processed_array)
                print(f"Processed image {i+1}/{len(image_paths)}: {image_path}")
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                print(f"Skipping image {i+1}/{len(image_paths)}")
                continue
        
        print(f"Successfully processed {len(processed_images)} out of {len(image_paths)} images")
        return processed_images


def process_single_image(image_path: str, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Convenience function to process a single image.
    
    Example from requirements - standalone implementation for reference.
    
    Args:
        image_path: Path to the image file
        target_size: Target dimensions (width, height)
        
    Returns:
        Numpy array (RGBA) of processed image
    """
    # Load and convert to RGBA
    img = Image.open(image_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Calculate resize dimensions maintaining aspect ratio
    width, height = img.size
    aspect = width / height
    target_width, target_height = target_size
    
    if aspect > 1:  # Wider than tall
        new_width = target_width
        new_height = int(target_width / aspect)
    else:  # Taller than wide or square
        new_height = target_height
        new_width = int(target_height * aspect)
    
    # Resize using LANCZOS filter
    img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create padded image with transparent background
    final = Image.new('RGBA', target_size, (0, 0, 0, 0))
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    final.paste(img, (offset_x, offset_y))
    
    return np.array(final)


if __name__ == "__main__":
    # Example usage and testing
    processor = ImageProcessor()
    
    # Test with a sample image (if available)
    try:
        result = processor.process_image("test_image.jpg")
        print(f"Processed image shape: {result.shape}")
        print(f"Data type: {result.dtype}")
        print("Image processing module is working correctly!")
        
    except FileNotFoundError:
        print("No test image found. Module is ready for use.")
    except Exception as e:
        print(f"Error during testing: {e}")
