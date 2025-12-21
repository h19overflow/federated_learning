"""
Data loading wrapper for comparative analysis.

Wraps existing data_processing utilities to provide a unified interface
for loading and preparing datasets for analysis experiments.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.utils import DataSourceExtractor
from federated_pneumonia_detection.src.utils.data_processing import (
    load_metadata,
    get_data_statistics,
    create_train_val_split,
    sample_dataframe,
)


class AnalysisDataLoader:
    """
    Unified data loader for comparative analysis experiments.

    Wraps existing data processing utilities to provide consistent
    data loading across centralized and federated experiments.
    """

    def __init__(
        self,
        source_path: str | Path,
        csv_filename: str = "stage2_train_metadata.csv",
        config: Optional[ConfigManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize data loader.

        Args:
            source_path: Path to ZIP file or directory containing dataset
            csv_filename: Metadata CSV filename
            config: Optional ConfigManager instance
            logger: Optional logger instance
        """
        self.source_path = Path(source_path)
        self.csv_filename = csv_filename
        self.config = config or ConfigManager()
        self.logger = logger or logging.getLogger(__name__)
        self._extractor = DataSourceExtractor(self.logger)
        self._image_dir: Optional[Path] = None
        self._csv_path: Optional[Path] = None
        self._df: Optional[pd.DataFrame] = None

    def extract_and_validate(self) -> Tuple[Path, Path]:
        """
        Extract dataset and validate contents.

        Returns:
            Tuple of (image_dir, csv_path)
        """
        try:
            image_dir, csv_path = self._extractor.extract_and_validate(
                str(self.source_path), self.csv_filename
            )
            self._image_dir = Path(image_dir)
            self._csv_path = Path(csv_path)
            self.logger.info(f"Dataset extracted: images={image_dir}, csv={csv_path}")
            return self._image_dir, self._csv_path
        except Exception as e:
            self.logger.error(f"AnalysisDataLoader:extract_and_validate - {type(e).__name__}: {e}")
            raise

    def load_full_dataset(self) -> pd.DataFrame:
        """
        Load complete dataset without sampling or splitting.

        Returns:
            Full DataFrame with all samples
        """
        if self._csv_path is None:
            self.extract_and_validate()

        try:
            self._df = load_metadata(str(self._csv_path), self.config, self.logger)
            self.logger.info(f"Loaded full dataset: {len(self._df)} samples")
            return self._df
        except Exception as e:
            self.logger.error(f"AnalysisDataLoader:load_full_dataset - {type(e).__name__}: {e}")
            raise

    def load_with_split(
        self,
        validation_split: float = 0.2,
        sample_fraction: float = 1.0,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load dataset with train/validation split.

        Args:
            validation_split: Fraction for validation set
            sample_fraction: Fraction of data to use (1.0 = all)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_df, val_df)
        """
        if self._df is None:
            self.load_full_dataset()

        target_col = self.config.get("columns.target")
        df = self._df.copy()

        try:
            if sample_fraction < 1.0:
                df = sample_dataframe(df, sample_fraction, target_col, seed, self.logger)

            train_df, val_df = create_train_val_split(
                df, validation_split, target_col, seed, self.logger
            )
            self.logger.info(f"Split complete: train={len(train_df)}, val={len(val_df)}")
            return train_df, val_df
        except Exception as e:
            self.logger.error(f"AnalysisDataLoader:load_with_split - {type(e).__name__}: {e}")
            raise

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        if self._df is None:
            self.load_full_dataset()

        target_col = self.config.get("columns.target")
        return get_data_statistics(self._df, target_col)

    @property
    def image_dir(self) -> Optional[Path]:
        """Get extracted image directory path."""
        return self._image_dir

    @property
    def csv_path(self) -> Optional[Path]:
        """Get CSV file path."""
        return self._csv_path

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """Get loaded DataFrame."""
        return self._df

    def cleanup(self) -> None:
        """Clean up temporary extraction directories."""
        try:
            self._extractor.cleanup()
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")
