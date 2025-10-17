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

import logging
from typing import Tuple, Optional, Union
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.utils.image_transforms import TransformBuilder
from federated_pneumonia_detection.src.entities.custom_image_dataset import CustomImageDataset


class ClientDataManager:
    """
    Manages DataLoader creation for federated clients.

    Handles train/validation splitting, transform creation, and dataset
    instantiation for individual client data partitions.
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        constants: SystemConstants,
        config: ExperimentConfig,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize client data manager.

        Args:
            image_dir: Directory containing image files
            constants: SystemConstants for configuration
            config: ExperimentConfig for parameters
            logger: Optional logger instance

        Raises:
            ValueError: If image_dir doesn't exist or is invalid
        """
        self.image_dir = Path(image_dir)
        self.constants = constants
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Validate image directory
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        if not self.image_dir.is_dir():
            raise ValueError(f"Image directory path is not a directory: {image_dir}")

        # Create transform builder once (expensive operation)
        self.transform_builder = TransformBuilder(constants, config)

    def create_dataloaders_for_partition(
        self,
        partition_df: pd.DataFrame,
        validation_split: Optional[float] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation DataLoaders from a client's data partition.

        Args:
            partition_df: DataFrame with 'filename' and 'Target' columns
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

        required_cols = [self.constants.FILENAME_COLUMN, self.constants.TARGET_COLUMN]
        missing_cols = [col for col in required_cols if col not in partition_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        val_split = validation_split or self.config.validation_split

        # Split partition into train and validation
        train_df, val_df = self._split_partition(partition_df, val_split)

        self.logger.info(
            f"Partition split: {len(train_df)} train, {len(val_df)} validation samples"
        )

        # Build transforms
        train_transform = self.transform_builder.build_training_transforms(
            enable_augmentation=True,
            augmentation_strength=self.config.augmentation_strength,
            custom_preprocessing=(
                self.config.get_custom_preprocessing_config()
                if self.config.use_custom_preprocessing
                else None
            )
        )

        val_transform = self.transform_builder.build_validation_transforms(
            custom_preprocessing=(
                self.config.get_custom_preprocessing_config()
                if self.config.use_custom_preprocessing
                else None
            )
        )

        # Create datasets
        try:
            train_dataset = CustomImageDataset(
                dataframe=train_df,
                image_dir=self.image_dir,
                constants=self.constants,
                transform=train_transform,
                color_mode=self.config.color_mode,
                validate_images=self.config.validate_images_on_init
            )

            val_dataset = CustomImageDataset(
                dataframe=val_df,
                image_dir=self.image_dir,
                constants=self.constants,
                transform=val_transform,
                color_mode=self.config.color_mode,
                validate_images=self.config.validate_images_on_init
            )
        except Exception as e:
            self.logger.error(f"Failed to create datasets: {e}")
            raise RuntimeError(f"Dataset creation failed: {e}") from e

        # Create DataLoaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows/Flower limitation
            drop_last=True,
            pin_memory=self.config.pin_memory,
            persistent_workers=False  # Windows compatibility
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # Windows/Flower limitation
            drop_last=False,
            pin_memory=self.config.pin_memory,
            persistent_workers=False  # Windows compatibility
        )

        self.logger.info(
            f"Created DataLoaders: {len(train_loader)} train batches, "
            f"{len(val_loader)} validation batches"
        )

        return train_loader, val_loader

    def _split_partition(
        self,
        partition_df: pd.DataFrame,
        validation_split: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split partition into train and validation sets.

        Attempts stratified split by target label. Falls back to random
        split if stratification fails (e.g., insufficient samples).

        Args:
            partition_df: DataFrame to split
            validation_split: Fraction for validation set

        Returns:
            Tuple of (train_df, val_df)
        """
        try:
            # Attempt stratified split by Target column
            train_df, val_df = train_test_split(
                partition_df,
                test_size=validation_split,
                stratify=partition_df[self.constants.TARGET_COLUMN],
                random_state=self.config.seed
            )
            self.logger.debug("Stratified split applied")
        except (ValueError, TypeError):
            # Fallback to random split if stratification fails
            self.logger.debug(
                "Stratification failed, falling back to random split"
            )
            train_df, val_df = train_test_split(
                partition_df,
                test_size=validation_split,
                random_state=self.config.seed
            )

        # Reset indices for consistency
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        return train_df, val_df
