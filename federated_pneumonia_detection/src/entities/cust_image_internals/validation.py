"""Validation utilities for CustomImageDataset."""

from typing import Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

logger = get_logger(__name__)


def validate_inputs(
    dataframe: pd.DataFrame,
    image_dir: Path,
    color_mode: str,
    filename_column: str,
    target_column: str,
) -> None:
    """
    Validate constructor inputs.

    Args:
        dataframe: DataFrame containing filename and target columns
        image_dir: Directory containing image files
        color_mode: Color mode ('RGB' or 'L')
        filename_column: Column name for filenames
        target_column: Column name for targets

    Raises:
        ValueError: If dataframe is invalid or required columns missing
        FileNotFoundError: If image directory doesn't exist
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe must be a pandas DataFrame")

    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not image_dir.is_dir():
        logger.error(f"Image directory path is not a directory: {image_dir}")
        raise ValueError(f"Image directory path is not a directory: {image_dir}")

    if color_mode not in ["RGB", "L"]:
        logger.error("Color mode must be 'RGB' or 'L'")
        raise ValueError("Color mode must be 'RGB' or 'L'")

    # Check required columns if dataframe is not empty
    if not dataframe.empty:
        required_columns = [filename_column, target_column]
        missing_columns = [
            col for col in required_columns if col not in dataframe.columns
        ]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")


def validate_image_files(filenames: np.ndarray, image_dir: Path) -> np.ndarray:
    """
    Validate that image files exist and are readable.

    Args:
        filenames: Array of image filenames
        image_dir: Directory containing images

    Returns:
        Array of valid indices
    """
    valid_indices = []
    invalid_count = 0

    for idx, filename in enumerate(filenames):
        image_path = image_dir / filename

        try:
            if not image_path.exists():
                logger.info(f"Image file not found: {image_path}")
                invalid_count += 1
                continue

            # Try to open the image to validate format
            with Image.open(image_path) as img:
                # Basic validation - ensure it's a valid image
                img.verify()

            valid_indices.append(idx)

        except Exception as e:
            logger.info(f"Invalid image file {image_path}: {e}")
            invalid_count += 1

    if invalid_count > 0:
        logger.info(
            f"Found {invalid_count} invalid image files out of {len(filenames)}"
        )

    return np.array(valid_indices)


def validate_all_images(filenames: np.ndarray, image_dir: Path) -> Tuple[int, int, list]:
    """
    Validate all images in the dataset.

    Args:
        filenames: Array of image filenames
        image_dir: Directory containing images

    Returns:
        Tuple of (valid_count, invalid_count, invalid_files)
    """
    invalid_files = []
    valid_count = 0

    for idx in range(len(filenames)):
        filename = filenames[idx]
        image_path = image_dir / filename

        try:
            if not image_path.exists():
                invalid_files.append((filename, "File not found"))
                continue

            with Image.open(image_path) as img:
                img.verify()

            valid_count += 1

        except Exception as e:
            invalid_files.append((filename, str(e)))
            logger.error(f"Error validating image {image_path}: {e}")

    invalid_count = len(invalid_files)
    return valid_count, invalid_count, invalid_files
