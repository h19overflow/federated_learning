"""
Essential data processing utilities for X-ray dataset metadata handling.
Provides CSV loading, train/validation splitting, and data preparation utilities.
"""

import os
import logging
from typing import Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger


def load_metadata(
    metadata_path: Union[str, Path],
    constants: SystemConstants,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load and prepare metadata CSV file.

    Args:
        metadata_path: Path to metadata CSV file
        constants: SystemConstants for column configuration
        logger: Optional logger instance

    Returns:
        Prepared DataFrame with filename column added

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    if logger is None:
        logger = get_logger(__name__)

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata: {len(df)} samples from {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to load metadata file: {e}")
        raise ValueError(f"Failed to load metadata file: {e}")

    # Validate required columns
    if constants.PATIENT_ID_COLUMN not in df.columns:
        logger.error(f"Missing column: {constants.PATIENT_ID_COLUMN}")
        raise ValueError(f"Missing column: {constants.PATIENT_ID_COLUMN}")

    if constants.TARGET_COLUMN not in df.columns:
        logger.error(f"Missing column: {constants.TARGET_COLUMN}")
        raise ValueError(f"Missing column: {constants.TARGET_COLUMN}")

    # Prepare filename column
    df = df.copy()
    df[constants.FILENAME_COLUMN] = (
        df[constants.PATIENT_ID_COLUMN].astype(str) + constants.IMAGE_EXTENSION
    )

    # Ensure target column is consistent type
    df[constants.TARGET_COLUMN] = df[constants.TARGET_COLUMN].astype(int).astype(str)

    # Basic validation
    if df.empty:
        logger.error("DataFrame is empty")
        raise ValueError("DataFrame is empty")

    # Check for missing values in critical columns
    required_columns = [
        constants.PATIENT_ID_COLUMN,
        constants.TARGET_COLUMN,
        constants.FILENAME_COLUMN,
    ]

    for col in required_columns:
        if df[col].isna().any():
            logger.error(f"Missing values found in column: {col}")
            raise ValueError(f"Missing values found in column: {col}")

    logger.info(f"Metadata prepared successfully: {len(df)} samples")
    return df


def sample_dataframe(
    df: pd.DataFrame,
    sample_fraction: float,
    target_column: str,
    seed: int = 42,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Sample DataFrame while maintaining class distribution.

    Args:
        df: Input DataFrame to sample
        sample_fraction: Fraction of data to sample (0.0 to 1.0)
        target_column: Name of target column for stratification
        seed: Random seed for reproducibility
        logger: Optional logger instance

    Returns:
        Sampled DataFrame

    Raises:
        ValueError: If sampling fails or parameters are invalid
    """
    if logger is None:
        logger = get_logger(__name__)

    if not 0.0 < sample_fraction <= 1.0:
        logger.error("sample_fraction must be between 0.0 and 1.0")
        raise ValueError("sample_fraction must be between 0.0 and 1.0") from ValueError

    if sample_fraction >= 1.0:
        logger.info("Using full dataset (sample_fraction >= 1.0)")
        return df.copy()

    try:
        unique_targets = df[target_column].unique()

        if len(unique_targets) > 1:
            # Stratified sampling to maintain class balance
            df_sample = (
                df.groupby(target_column)
                .sample(frac=sample_fraction, random_state=seed)
                .reset_index(drop=True)
            )
            logger.info(f"Stratified sampling: {len(df_sample)} samples")
        else:
            # Simple random sampling for single class
            df_sample = df.sample(frac=sample_fraction, random_state=seed).reset_index(
                drop=True
            )
            logger.info(f"Random sampling: {len(df_sample)} samples")

        return df_sample

    except Exception as e:
        logger.error(f"Data sampling failed: {e}")
        raise ValueError(f"Data sampling failed: {e}")


def create_train_val_split(
    df: pd.DataFrame,
    validation_split: float,
    target_column: str,
    seed: int = 42,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation split with stratification when possible.

    Args:
        df: DataFrame to split
        validation_split: Fraction for validation (0.0 to 1.0)
        target_column: Name of target column for stratification
        seed: Random seed for reproducibility
        logger: Optional logger instance

    Returns:
        Tuple of (train_df, val_df) DataFrames

    Raises:
        ValueError: If split fails or parameters are invalid
    """
    if logger is None:
        logger = get_logger(__name__)

    if not 0.0 < validation_split < 1.0:
        logger.error("validation_split must be between 0.0 and 1.0")
        raise ValueError("validation_split must be between 0.0 and 1.0")

    try:
        unique_targets = df[target_column].unique()

        # Use stratification if multiple classes exist
        stratify = df[target_column] if len(unique_targets) > 1 else None

        train_df, val_df = train_test_split(
            df, test_size=validation_split, random_state=seed, stratify=stratify
        )

        logger.info(
            f"Created train/val split - Train: {len(train_df)}, Val: {len(val_df)}"
        )
        return train_df, val_df

    except Exception as e:
        logger.error(f"Train/validation split failed: {e}")
        raise ValueError(f"Train/validation split failed: {e}")


def load_and_split_data(
    constants: SystemConstants,
    config: ExperimentConfig,
    metadata_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete data loading and splitting pipeline.

    Args:
        constants: SystemConstants for configuration
        config: ExperimentConfig for processing parameters
        metadata_path: Optional custom metadata path (uses constants if None)

    Returns:
        Tuple of (train_df, val_df) DataFrames

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If data processing fails
    """
    logger = logging.getLogger(__name__)

    try:
        # Determine metadata path
        if metadata_path is None:
            metadata_path = os.path.join(
                constants.BASE_PATH, constants.METADATA_FILENAME
            )

        # Load metadata
        df = load_metadata(metadata_path, constants, logger)

        # Sample data if needed
        df_sample = sample_dataframe(
            df, config.sample_fraction, constants.TARGET_COLUMN, config.seed, logger
        )

        # Create train/validation split
        train_df, val_df = create_train_val_split(
            df_sample,
            config.validation_split,
            constants.TARGET_COLUMN,
            config.seed,
            logger,
        )

        logger.info(
            f"Data processing completed: {len(train_df)} train, {len(val_df)} validation samples"
        )
        return train_df, val_df

    except Exception as e:
        logger.error(f"Data processing pipeline failed: {e}")
        raise ValueError(f"Data processing pipeline failed: {e}") from e


def validate_image_paths(
    constants: SystemConstants, logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate that image directories exist.

    Args:
        constants: SystemConstants containing path configuration
        logger: Optional logger instance

    Returns:
        True if paths exist, False otherwise
    """
    if logger is None:
        logger = get_logger(__name__)

    main_images_path = os.path.join(constants.BASE_PATH, constants.MAIN_IMAGES_FOLDER)
    image_dir_path = os.path.join(main_images_path, constants.IMAGES_SUBFOLDER)

    if not os.path.exists(main_images_path):
        logger.error(f"Main images folder not found: {main_images_path}")
        return False

    if not os.path.exists(image_dir_path):
        logger.error(f"Images directory not found: {image_dir_path}")
        return False

    logger.info("Image paths validated successfully")
    return True


def get_image_directory_path(constants: SystemConstants) -> str:
    """
    Get the full path to the image directory.

    Args:
        constants: SystemConstants containing path configuration

    Returns:
        Full path to image directory
    """
    logger = get_logger(__name__)
    main_images_path = os.path.join(constants.BASE_PATH, constants.MAIN_IMAGES_FOLDER)
    return os.path.join(main_images_path, constants.IMAGES_SUBFOLDER)


def get_data_statistics(df: pd.DataFrame, target_column: str) -> dict:
    """
    Get basic statistics about a DataFrame.

    Args:
        df: DataFrame to analyze
        target_column: Name of target column

    Returns:
        Dictionary with statistics
    """
    logger = get_logger(__name__)
    stats = {
        "total_samples": len(df),
        "class_distribution": df[target_column].value_counts().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "columns": list(df.columns),
    }

    if len(df) > 0:
        class_counts = df[target_column].value_counts()
        stats["class_balance_ratio"] = (
            class_counts.min() / class_counts.max() if len(class_counts) > 1 else 1.0
        )

    return stats


# Convenience wrapper for backward compatibility
class DataProcessor:
    """
    Lightweight wrapper around data processing functions for backward compatibility.

    DEPRECATED: Use the standalone functions instead.
    """

    def __init__(self, constants: SystemConstants):
        """Initialize with system constants."""
        self.constants = constants
        self.logger = get_logger(__name__)
        self.logger.warning(
            "DataProcessor class is deprecated. "
            "Use load_and_split_data() function instead."
        )

    def load_and_process_data(
        self, config: ExperimentConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process data using the new function-based approach."""
        return load_and_split_data(self.constants, config)

    def validate_image_paths(self) -> bool:
        """Validate image paths."""
        return validate_image_paths(self.constants, self.logger)

    def get_image_paths(self) -> Tuple[str, str]:
        """Get image paths."""
        main_images_path = os.path.join(
            self.constants.BASE_PATH, self.constants.MAIN_IMAGES_FOLDER
        )
        image_dir_path = get_image_directory_path(self.constants)
        return main_images_path, image_dir_path

    # Private methods for backward compatibility with tests
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata (backward compatibility wrapper)."""
        metadata_path = os.path.join(
            self.constants.BASE_PATH, self.constants.METADATA_FILENAME
        )
        return load_metadata(metadata_path, self.constants, self.logger)

    def _prepare_filenames(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare filenames (backward compatibility wrapper)."""
        df = df.copy()
        if self.constants.PATIENT_ID_COLUMN not in df.columns:
            raise ValueError(f"Missing column: {self.constants.PATIENT_ID_COLUMN}")
        df[self.constants.FILENAME_COLUMN] = (
            df[self.constants.PATIENT_ID_COLUMN].astype(str)
            + self.constants.IMAGE_EXTENSION
        )
        df[self.constants.TARGET_COLUMN] = (
            df[self.constants.TARGET_COLUMN].astype(int).astype(str)
        )
        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate data (backward compatibility wrapper)."""
        if df.empty:
            self.logger.error("DataFrame is empty")
            raise ValueError("DataFrame is empty")

        required_columns = [
            self.constants.PATIENT_ID_COLUMN,
            self.constants.TARGET_COLUMN,
            self.constants.FILENAME_COLUMN,
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        for col in required_columns:
            if df[col].isna().any():
                self.logger.error(f"Missing values found in column: {col}")
                raise ValueError(f"Missing values found in column: {col}")

    def _sample_data(
        self, df: pd.DataFrame, sample_fraction: float, seed: int
    ) -> pd.DataFrame:
        """Sample data (backward compatibility wrapper)."""
        return sample_dataframe(
            df, sample_fraction, self.constants.TARGET_COLUMN, seed, self.logger
        )

    def _create_train_val_split(
        self, df: pd.DataFrame, validation_split: float, seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/val split (backward compatibility wrapper)."""
        return create_train_val_split(
            df, validation_split, self.constants.TARGET_COLUMN, seed, self.logger
        )
