"""
Core data processing utility functions for X-ray dataset metadata handling.

This module provides standalone functions for:
- CSV loading and metadata preparation
- Train/validation splitting with stratification
- Data sampling while maintaining class distribution
- Image path validation and retrieval
- Data statistics computation

These functions are designed to work with both the new ConfigManager
and legacy configuration objects for backward compatibility.
"""

import os
import logging
import warnings
from typing import Tuple, Optional, Union, TYPE_CHECKING
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager


def load_metadata(
    metadata_path: Union[str, Path],
    config_or_constants: Optional[Union['ConfigManager']] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load and prepare metadata CSV file.

    Args:
        metadata_path: Path to metadata CSV file
            config_or_constants: ConfigManager
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

    # Handle both ConfigManager and old SystemConstants
    if config_or_constants is None:
        from federated_pneumonia_detection.config.config_manager import ConfigManager
        config = ConfigManager()
    elif hasattr(config_or_constants, 'get'):  # ConfigManager
        config = config_or_constants
    else:  # Old SystemConstants
        warnings.warn(
            "Passing SystemConstants to load_metadata is deprecated. "
            "Use ConfigManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Convert old constants to config access pattern
        constants = config_or_constants
        from federated_pneumonia_detection.config.config_manager import ConfigManager
        config = ConfigManager()
        
        patient_id_col = constants.PATIENT_ID_COLUMN
        target_col = constants.TARGET_COLUMN
        filename_col = constants.FILENAME_COLUMN
        image_ext = constants.IMAGE_EXTENSION

    # New ConfigManager path
    patient_id_col = config.get('columns.patient_id')
    target_col = config.get('columns.target')
    filename_col = config.get('columns.filename')
    image_ext = config.get('system.image_extension')

    # Validate required columns
    if patient_id_col not in df.columns:
        logger.error(f"Missing column: {patient_id_col}")
        raise ValueError(f"Missing column: {patient_id_col}")

    if target_col not in df.columns:
        logger.error(f"Missing column: {target_col}")
        raise ValueError(f"Missing column: {target_col}")

    # Prepare filename column
    df = df.copy()
    df[filename_col] = df[patient_id_col].astype(str) + image_ext
    df[target_col] = df[target_col].astype(int).astype(str)

    # Basic validation
    if df.empty:
        logger.error("DataFrame is empty")
        raise ValueError("DataFrame is empty")

    # Check for missing values in critical columns
    required_columns = [patient_id_col, target_col, filename_col]

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
    config_or_constants: Optional[Union['ConfigManager']] = None,
    metadata_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete data loading and splitting pipeline.

    Args:
        config_or_constants: ConfigManager (new) or SystemConstants (deprecated)
        config_or_experiment: ConfigManager (new) or ExperimentConfig (deprecated)
        metadata_path: Optional custom metadata path

    Returns:
        Tuple of (train_df, val_df) DataFrames

    Raises:
        FileNotFoundError: If metadata file doesn't exist
        ValueError: If data processing fails
    """
    logger = get_logger(__name__)

    try:
        # Handle ConfigManager vs old objects
        if config_or_constants is None:
            from federated_pneumonia_detection.config.config_manager import ConfigManager
            config = ConfigManager()
        elif hasattr(config_or_constants, 'get'):  # ConfigManager
            config = config_or_constants
        else:  # Old SystemConstants (deprecated but supported via load_metadata)
            config = None # load_metadata will handle converting constants

        # New ConfigManager path
        if metadata_path is None:
            if config:
                base_path = config.get('paths.base_path', '.')
                metadata_filename = config.get('paths.metadata_filename', 'Train_metadata.csv')
            else:
                # SystemConstants path
                base_path = config_or_constants.BASE_PATH
                metadata_filename = config_or_constants.METADATA_FILENAME
            
            metadata_path = os.path.join(base_path, metadata_filename)

        df = load_metadata(metadata_path, config_or_constants, logger)
        
        if config:
            sample_frac = config.get('experiment.sample_fraction', 0.1)
            val_split = config.get('experiment.validation_split', 0.2)
            seed = config.get('experiment.seed', 42)
            target_column = config.get('columns.target', 'Target')
        else:
            # Fallback for deprecated constants
            from federated_pneumonia_detection.config.config_manager import ConfigManager
            config_fallback = ConfigManager()
            sample_frac = config_fallback.get('experiment.sample_fraction', 0.1)
            val_split = config_fallback.get('experiment.validation_split', 0.2)
            seed = config_fallback.get('experiment.seed', 42)
            target_column = config_or_constants.TARGET_COLUMN

        df_sample = sample_dataframe(
            df, sample_frac, target_column, seed, logger
        )

        train_df, val_df = create_train_val_split(
            df_sample,
            val_split,
            target_column,
            seed,
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
    config_or_constants: Optional[Union['ConfigManager']] = None,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Validate that image directories exist.

    Args:
        config_or_constants: ConfigManager (new) or SystemConstants (deprecated)
        logger: Optional logger instance

    Returns:
        True if paths exist, False otherwise
    """
    if logger is None:
        logger = get_logger(__name__)

    if config_or_constants is None:
        from federated_pneumonia_detection.config.config_manager import ConfigManager
        config = ConfigManager()
    elif hasattr(config_or_constants, 'get'):  # ConfigManager
        config = config_or_constants
    else:  # Old SystemConstants
        constants = config_or_constants
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

    # New ConfigManager path
    base_path = config.get('paths.base_path', '.')
    main_images_folder = config.get('paths.main_images_folder', 'Images')
    images_subfolder = config.get('paths.images_subfolder', 'Images')
    
    main_images_path = os.path.join(base_path, main_images_folder)
    image_dir_path = os.path.join(main_images_path, images_subfolder)

    if not os.path.exists(main_images_path):
        logger.error(f"Main images folder not found: {main_images_path}")
        return False

    if not os.path.exists(image_dir_path):
        logger.error(f"Images directory not found: {image_dir_path}")
        return False

    logger.info("Image paths validated successfully")
    return True


def get_image_directory_path(
    config_or_constants: Optional[Union['ConfigManager']] = None
) -> str:
    """
    Get the full path to the image directory.

    Args:
        config_or_constants: ConfigManager (new) or SystemConstants (deprecated)

    Returns:
        Full path to image directory
    """
    
    if config_or_constants is None:
        from federated_pneumonia_detection.config.config_manager import ConfigManager
        config = ConfigManager()
    elif hasattr(config_or_constants, 'get'):  # ConfigManager
        config = config_or_constants
    else:  # Old SystemConstants
        constants = config_or_constants
        main_images_path = os.path.join(constants.BASE_PATH, constants.MAIN_IMAGES_FOLDER)
        return os.path.join(main_images_path, constants.IMAGES_SUBFOLDER)

    # New ConfigManager path
    base_path = config.get('paths.base_path', '.')
    main_images_folder = config.get('paths.main_images_folder', 'Images')
    images_subfolder = config.get('paths.images_subfolder', 'Images')
    
    main_images_path = os.path.join(base_path, main_images_folder)
    return os.path.join(main_images_path, images_subfolder)


def get_data_statistics(df: pd.DataFrame, target_column: str) -> dict:
    """
    Get basic statistics about a DataFrame.

    Args:
        df: DataFrame to analyze
        target_column: Name of target column

    Returns:
        Dictionary with statistics
    """
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
