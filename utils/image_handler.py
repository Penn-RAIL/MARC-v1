"""
ImageHandler - Utility for loading and processing images
"""
from PIL import Image
import io
from pathlib import Path
from typing import Optional


class ImageHandler:
    """Handle image loading, validation, and preprocessing"""
    
    # Supported image formats
    SUPPORTED_FORMATS = {'jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp'}
    
    # Maximum file size (5MB)
    MAX_SIZE_BYTES = 5 * 1024 * 1024
    
    # Maximum dimension for resizing
    MAX_DIMENSION = 2048
    
    @staticmethod
    def load_image(file_path: str) -> bytes:
        """
        Load an image file and return as bytes
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Image data as bytes
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid or too large
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Check file extension
        extension = path.suffix.lower().lstrip('.')
        if extension not in ImageHandler.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {extension}. "
                f"Supported formats: {', '.join(ImageHandler.SUPPORTED_FORMATS)}"
            )
        
        # Load file
        with open(file_path, 'rb') as f:
            image_data = f.read()
        
        # Check file size
        if len(image_data) > ImageHandler.MAX_SIZE_BYTES:
            raise ValueError(
                f"Image too large: {len(image_data)} bytes. "
                f"Maximum size: {ImageHandler.MAX_SIZE_BYTES} bytes"
            )
        
        # Validate it's a real image
        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")
        
        return image_data
    
    @staticmethod
    def resize_if_needed(
        image_data: bytes, 
        max_dimension: Optional[int] = None
    ) -> bytes:
        """
        Resize image if it exceeds maximum dimension
        
        Args:
            image_data: Original image bytes
            max_dimension: Maximum width or height (default: MAX_DIMENSION)
            
        Returns:
            Resized image bytes (or original if no resize needed)
        """
        if max_dimension is None:
            max_dimension = ImageHandler.MAX_DIMENSION
        
        # Open image
        img = Image.open(io.BytesIO(image_data))
        
        # Check if resize needed
        if max(img.size) <= max_dimension:
            return image_data
        
        # Calculate new size maintaining aspect ratio
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        
        # Save to bytes
        output = io.BytesIO()
        
        # Preserve original format or use JPEG as fallback
        format_to_save = img.format if img.format else 'JPEG'
        
        # Convert RGBA to RGB if saving as JPEG
        if format_to_save == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = rgb_img
        
        img.save(output, format=format_to_save, quality=85)
        return output.getvalue()
    
    @staticmethod
    def get_image_info(image_data: bytes) -> dict:
        """
        Get information about an image
        
        Args:
            image_data: Image bytes
            
        Returns:
            Dictionary with image metadata
        """
        img = Image.open(io.BytesIO(image_data))
        
        return {
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'width': img.width,
            'height': img.height,
            'file_size_bytes': len(image_data)
        }