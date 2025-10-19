"""
Client data management for federated learning.

Creates and manages DataLoaders for individual client partitions,
handling transforms, dataset creation, and train/val splits.

Dependencies:
- TransformBuilder: Image augmentation
- CustomImageDataset: Dataset class
- PyTorch DataLoader: Batch loading
- sklearn.model_selection: Stratified splitting

Role in System:
- Converts DataFrame partitions into PyTorch DataLoaders
- Applies appropriate transforms for training and validation
- Handles stratified train/val splitting
- Manages dataset creation with error handling
"""

from typing import Tuple, Optional, Union
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.utils.image_transforms import TransformBuilder
from federated_pneumonia_detection.src.entities.custom_image_dataset import (
    CustomImageDataset,
)


def split_partition(
    partition_df: pd.DataFrame,
    validation_split: float,
    target_column: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split partition into train and validation sets.

    Attempts stratified split by target label. Falls back to random
    split if stratification fails (e.g., insufficient samples).

    Args:
        partition_df: DataFrame to split
        validation_split: Fraction for validation set
        target_column: Name of the target column
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df)
    """
    try:
        # Attempt stratified split by target column
        train_df, val_df = train_test_split(
            partition_df,
            test_size=validation_split,
            stratify=partition_df[target_column],
            random_state=seed,
        )
    except (ValueError, TypeError):
        # Fallback to random split if stratification fails
        train_df, val_df = train_test_split(
            partition_df,
            test_size=validation_split,
            random_state=seed,
        )

    # Reset indices for consistency
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df


def load_data(
    partition_df: pd.DataFrame,
    image_dir: Union[str, Path],
    constants: SystemConstants,
    config: ExperimentConfig,
    validation_split: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from a client's data partition.

    Args:
        partition_df: DataFrame with filename and target columns
        image_dir: Directory containing image files
        constants: SystemConstants for configuration
        config: ExperimentConfig for parameters
        validation_split: Fraction for validation split. Defaults to config.validation_split

    Returns:
        Tuple of (train_loader, validation_loader)

    Raises:
        ValueError: If partition_df is empty or missing required columns
        RuntimeError: If dataset creation fails
    """
    # Validate inputs
    if partition_df.empty:
        raise ValueError("partition_df cannot be empty")

    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image directory not found: {image_dir}")

    required_cols = [constants.FILENAME_COLUMN, constants.TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in partition_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    val_split = validation_split or config.validation_split

    # Split partition into train and validation
    train_df, val_df = split_partition(
        partition_df,
        val_split,
        constants.TARGET_COLUMN,
        config.seed,
    )

    # Build transforms
    transform_builder = TransformBuilder(constants, config)

    train_transform = transform_builder.build_training_transforms(
        enable_augmentation=True,
        augmentation_strength=config.augmentation_strength,
        custom_preprocessing=(
            config.get_custom_preprocessing_config()
            if config.use_custom_preprocessing
            else None
        ),
    )

    val_transform = transform_builder.build_validation_transforms(
        custom_preprocessing=(
            config.get_custom_preprocessing_config()
            if config.use_custom_preprocessing
            else None
        ),
    )

    # Create datasets
    try:
        train_dataset = CustomImageDataset(
            dataframe=train_df,
            image_dir=image_dir,
            constants=constants,
            transform=train_transform,
            color_mode=config.color_mode,
            validate_images=config.validate_images_on_init,
        )

        val_dataset = CustomImageDataset(
            dataframe=val_df,
            image_dir=image_dir,
            constants=constants,
            transform=val_transform,
            color_mode=config.color_mode,
            validate_images=config.validate_images_on_init,
        )
    except Exception as e:
        raise RuntimeError(f"Dataset creation failed: {e}") from e

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows/Flower limitation
        drop_last=True,
        pin_memory=config.pin_memory,
        persistent_workers=False,  # Windows compatibility
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Windows/Flower limitation
        drop_last=False,
        pin_memory=config.pin_memory,
        persistent_workers=False,  # Windows compatibility
    )

    return train_loader, val_loader
