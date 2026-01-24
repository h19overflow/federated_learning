"""Data preparation utilities for training."""

import logging
from typing import Tuple

import pandas as pd

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.internals.data.xray_data_module import (
    XRayDataModule,
)
from federated_pneumonia_detection.src.internals.data_processing import (
    create_train_val_split,
    load_metadata,
)


def prepare_dataset(
    csv_path: str,
    image_dir: str,
    config: ConfigManager,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare training dataset.

    Args:
        csv_path: Path to metadata CSV file
        image_dir: Directory containing images
        config: Configuration manager instance
        logger: Logger instance

    Returns:
        Tuple of (train_df, val_df)
    """
    df = load_metadata(csv_path, config, logger)
    logger.info(f"Loaded metadata: {len(df)} samples from {csv_path}")

    target_col = config.get("columns.target")
    train_df, val_df = create_train_val_split(
        df,
        config.get("experiment.validation_split"),
        target_col,
        config.get("experiment.seed"),
        logger,
    )

    logger.info(f"Dataset prepared: {len(train_df)} train, {len(val_df)} validation")
    return train_df, val_df


def create_data_module(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: str,
    config: ConfigManager,
    logger: logging.Logger,
) -> XRayDataModule:
    """
    Create PyTorch Lightning DataModule.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        image_dir: Directory containing images
        config: Configuration manager instance
        logger: Logger instance

    Returns:
        XRayDataModule instance
    """
    logger.info("Setting up data module...")

    data_module = XRayDataModule(
        train_df=train_df,
        val_df=val_df,
        config=config,
        image_dir=image_dir,
        validate_images_on_init=False,
    )

    return data_module
