"""Image loading operations for CustomImageDataset."""

from pathlib import Path
from PIL import Image
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

logger = get_logger(__name__)


def load_image(filename: str, image_dir: Path, color_mode: str) -> Image.Image:
    """
    Load and convert image with error handling.

    Args:
        filename: Name of image file to load
        image_dir: Directory containing images
        color_mode: Target color mode ('RGB' or 'L')

    Returns:
        PIL Image in specified color mode

    Raises:
        RuntimeError: If image loading fails
    """
    image_path = image_dir / filename

    try:
        with Image.open(image_path) as img:
            # Convert to RGB first to handle various input formats
            img_rgb = img.convert("RGB")

            # Then convert to target mode if different
            if color_mode == "L":
                return img_rgb.convert("L")
            else:
                return img_rgb

    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise RuntimeError(f"Failed to load image {image_path}: {e}") from e
