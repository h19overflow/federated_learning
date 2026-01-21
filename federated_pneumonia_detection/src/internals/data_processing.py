"""
Data processing orchestrator module.

This module provides the DataProcessor class which orchestrates data processing workflows
by coordinating the utility functions from data_processing_functions.

Architecture:
- data_processing_functions.py: Contains pure utility functions (the logic)
- DataProcessor class: Orchestrates and coordinates these functions (the orchestrator)

For direct function usage, import from data_processing_functions.
For orchestrated workflows, use the DataProcessor class.
"""

import os
from typing import Tuple, TYPE_CHECKING
import pandas as pd

from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

# Import all utility functions from the new module
from federated_pneumonia_detection.src.internals.data_processing_functions import (
    load_metadata,
    sample_dataframe,
    create_train_val_split,
    load_and_split_data,
    validate_image_paths,
    get_image_directory_path,
    get_data_statistics,
)

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager


# Re-export all functions for backward compatibility
__all__ = [
    "load_metadata",
    "sample_dataframe",
    "create_train_val_split",
    "load_and_split_data",
    "validate_image_paths",
    "get_image_directory_path",
    "get_data_statistics",
    "DataProcessor",
]


class DataProcessor:
    """
    Data processing orchestrator for pneumonia detection workflows.

    This class orchestrates data processing operations by coordinating the utility
    functions from data_processing_functions. It provides a high-level interface
    for common data processing workflows while maintaining state and configuration.
    """

    def __init__(self, config: "ConfigManager"):
        """
        Initialize with config.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.logger = get_logger(__name__)

    def load_and_process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and process data using the orchestrated workflow.

        Returns:
            Tuple of (train_df, val_df) DataFrames
        """
        return load_and_split_data(self.config)

    def validate_image_paths(self) -> bool:
        """
        Validate image paths.

        Returns:
            True if paths exist, False otherwise
        """
        return validate_image_paths(self.config, self.logger)

    def get_image_paths(self) -> Tuple[str, str]:
        """
        Get image paths.

        Returns:
            Tuple of (main_images_path, image_dir_path)
        """
        main_images_path = os.path.join(
            self.config.get("paths.base_path", "."),
            self.config.get("paths.main_images_folder", "Images"),
        )
        image_dir_path = get_image_directory_path(self.config)
        return main_images_path, image_dir_path

    # Private methods for backward compatibility with tests
    def _load_metadata(self) -> pd.DataFrame:
        """
        Load metadata (backward compatibility wrapper).

        Returns:
            Prepared DataFrame with metadata
        """
        metadata_path = os.path.join(
            self.config.get("paths.base_path", "."),
            self.config.get("paths.metadata_filename", "Train_metadata.csv"),
        )
        return load_metadata(metadata_path, self.config, self.logger)

    def _prepare_filenames(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare filenames (backward compatibility wrapper).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with prepared filenames

        Raises:
            ValueError: If required columns are missing
        """
        df = df.copy()
        if self.config.get("columns.patient_id") not in df.columns:
            raise ValueError(f"Missing column: {self.config.get('columns.patient_id')}")
        df[self.config.get("columns.filename")] = df[
            self.config.get("columns.patient_id")
        ].astype(str) + self.config.get("system.image_extension")
        df[self.config.get("columns.target")] = (
            df[self.config.get("columns.target")].astype(int).astype(str)
        )
        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate data (backward compatibility wrapper).

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If data validation fails
        """
        if df.empty:
            self.logger.error("DataFrame is empty")
            raise ValueError("DataFrame is empty")

        required_columns = [
            self.config.get("columns.patient_id"),
            self.config.get("columns.target"),
            self.config.get("columns.filename"),
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
        """
        Sample data (backward compatibility wrapper).

        Args:
            df: DataFrame to sample
            sample_fraction: Fraction of data to sample
            seed: Random seed

        Returns:
            Sampled DataFrame
        """
        return sample_dataframe(
            df, sample_fraction, self.config.get("columns.target"), seed, self.logger
        )

    def _create_train_val_split(
        self, df: pd.DataFrame, validation_split: float, seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/val split (backward compatibility wrapper).

        Args:
            df: DataFrame to split
            validation_split: Validation split fraction
            seed: Random seed

        Returns:
            Tuple of (train_df, val_df) DataFrames
        """
        return create_train_val_split(
            df, validation_split, self.config.get("columns.target"), seed, self.logger
        )
