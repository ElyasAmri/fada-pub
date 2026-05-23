"""
Image Processing Utilities
Shared image conversion and processing functions for VLM APIs
"""

import io
import base64
from pathlib import Path
from typing import Union
from PIL import Image


def to_jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    """
    Convert PIL Image to JPEG bytes.

    Args:
        image: PIL Image object
        quality: JPEG quality (1-100)

    Returns:
        JPEG image as bytes
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    return buffer.getvalue()


def to_base64(image: Image.Image, quality: int = 95) -> str:
    """
    Convert PIL Image to base64-encoded JPEG string.

    Args:
        image: PIL Image object
        quality: JPEG quality (1-100)

    Returns:
        Base64-encoded string
    """
    jpeg_bytes = to_jpeg_bytes(image, quality)
    return base64.b64encode(jpeg_bytes).decode('utf-8')


def to_base64_data_url(image: Image.Image, quality: int = 95) -> str:
    """
    Convert PIL Image to base64 data URL for API use.

    Args:
        image: PIL Image object
        quality: JPEG quality (1-100)

    Returns:
        Data URL string (data:image/jpeg;base64,...)
    """
    b64_string = to_base64(image, quality)
    return f"data:image/jpeg;base64,{b64_string}"


def load_and_convert(path: Union[str, Path]) -> Image.Image:
    """
    Load image from path and convert to RGB.

    Args:
        path: Path to image file

    Returns:
        PIL Image in RGB mode

    Raises:
        FileNotFoundError: If image file doesn't exist
        PIL.UnidentifiedImageError: If file is not a valid image
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image
