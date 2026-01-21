"""Image format conversion utilities for ReportLab."""

import base64
import io
import logging
from typing import Optional

from PIL import Image
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage

logger = logging.getLogger(__name__)


def base64_to_image(
    base64_string: str, max_width: float = 4 * inch
) -> Optional[RLImage]:
    """Convert base64 string to ReportLab Image.

    Args:
        base64_string: Base64 encoded image data
        max_width: Maximum width for the image

    Returns:
        ReportLab Image object or None if conversion fails
    """
    try:
        image_data = base64.b64decode(base64_string)
        image_buffer = io.BytesIO(image_data)

        pil_image = Image.open(image_buffer)
        width, height = pil_image.size
        aspect_ratio = height / width

        image_buffer.seek(0)

        img_width = min(max_width, 4 * inch)
        img_height = img_width * aspect_ratio

        return RLImage(image_buffer, width=img_width, height=img_height)
    except Exception as e:
        logger.error(f"Failed to convert base64 to image: {e}")
        return None


def pil_to_reportlab_image(
    pil_image: Image.Image, max_width: float = 4 * inch
) -> Optional[RLImage]:
    """Convert PIL Image to ReportLab Image.

    Args:
        pil_image: PIL Image object
        max_width: Maximum width for the image

    Returns:
        ReportLab Image object or None if conversion fails
    """
    try:
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)

        width, height = pil_image.size
        aspect_ratio = height / width

        img_width = min(max_width, 4 * inch)
        img_height = img_width * aspect_ratio

        return RLImage(buffer, width=img_width, height=img_height)
    except Exception as e:
        logger.error(f"Failed to convert PIL image: {e}")
        return None
