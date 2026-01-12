"""Input validation utilities for XRayDataModule."""

from pathlib import Path
import pandas as pd
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

logger = get_logger(__name__)


def validate_inputs(
    image_dir: Path,
    color_mode: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    filename_column: str,
    target_column: str,
) -> None:
    """
    Validate XRayDataModule initialization inputs.

    Args:
        image_dir: Directory containing image files
        color_mode: Color mode ('RGB' or 'L')
        train_df: Training DataFrame
        val_df: Validation DataFrame
        filename_column: Column name for filenames
        target_column: Column name for targets

    Raises:
        FileNotFoundError: If image directory doesn't exist
        ValueError: If invalid inputs detected
    """
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not image_dir.is_dir():
        logger.error(f"Image directory path is not a directory: {image_dir}")
        raise ValueError(f"Image directory path is not a directory: {image_dir}")

    if color_mode not in ['RGB', 'L']:
        logger.error("color_mode must be 'RGB' or 'L'")
        raise ValueError("color_mode must be 'RGB' or 'L'")

    if train_df.empty and val_df.empty:
        logger.error("Both train and validation DataFrames are empty")
        raise ValueError("Both train and validation DataFrames are empty")

    # Validate required columns
    for name, df in [('train', train_df), ('val', val_df)]:
        if not df.empty:
            required_cols = [filename_column, target_column]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns in {name} DataFrame: {missing_cols}")
                raise ValueError(f"Missing columns in {name} DataFrame: {missing_cols}")
