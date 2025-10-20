"""
PyTorch Lightning DataModule for X-ray image data management.
Orchestrates dataset creation, data loading, and batch management with comprehensive configuration.
"""

import logging
from typing import Optional, Union, Dict, Any
from pathlib import Path

import torch
import pandas as pd
import pytorch_lightning as pl
from numpy import ndarray, dtype
from torch.utils.data import DataLoader
import numpy as np

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.entities.custom_image_dataset import CustomImageDataset
from federated_pneumonia_detection.src.utils.image_transforms import TransformBuilder


class XRayDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for X-ray image data management.

    Handles dataset creation, data loading, and batch management with
    comprehensive configuration and error handling.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        constants: SystemConstants,
        config: ExperimentConfig,
        image_dir: Union[str, Path],
        test_df: Optional[pd.DataFrame] = None,
        color_mode: str = 'RGB',
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        validate_images_on_init: bool = True,
        custom_preprocessing_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the XRayDataModule.

        Args:
            train_df: DataFrame containing training data
            val_df: DataFrame containing validation data
            constants: SystemConstants for configuration
            config: ExperimentConfig for parameters
            image_dir: Directory containing image files
            test_df: Optional DataFrame for test data
            color_mode: 'RGB' or 'L' for color mode
            pin_memory: Whether to use pinned memory for DataLoader
            persistent_workers: Whether to keep workers alive between epochs
            prefetch_factor: Number of batches to prefetch per worker
            validate_images_on_init: Whether to validate images during initialization
            custom_preprocessing_config: Custom preprocessing parameters
        """
        super().__init__()

        self.save_hyperparameters(ignore=['train_df', 'val_df', 'test_df'])

        # Store configuration
        self.constants = constants
        self.config = config
        self.train_df = train_df.copy() if not train_df.empty else train_df
        self.val_df = val_df.copy() if not val_df.empty else val_df
        self.test_df = test_df.copy() if test_df is not None and not test_df.empty else None
        self.image_dir = Path(image_dir)
        self.color_mode = color_mode.upper()
        self.validate_images_on_init = validate_images_on_init

        # DataLoader configuration
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers and config.num_workers > 0
        self.prefetch_factor = prefetch_factor if config.num_workers > 0 else 2

        # Custom preprocessing
        self.custom_preprocessing_config = custom_preprocessing_config or {}

        # Initialize components
        self.logger = logging.getLogger(__name__)
        self.transform_builder = TransformBuilder(constants, config)

        # Datasets (will be created in setup())
        self.train_dataset: Optional[CustomImageDataset] = None
        self.val_dataset: Optional[CustomImageDataset] = None
        self.test_dataset: Optional[CustomImageDataset] = None

        # Validate inputs
        self._validate_inputs()

        self.logger.info(f"DataModule initialized - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df) if test_df is not None else 0}")

    def _validate_inputs(self) -> None:
        """Validate initialization inputs."""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if not self.image_dir.is_dir():
            raise ValueError(f"Image directory path is not a directory: {self.image_dir}")

        if self.color_mode not in ['RGB', 'L']:
            raise ValueError("color_mode must be 'RGB' or 'L'")

        if self.train_df.empty and self.val_df.empty:
            raise ValueError("Both train and validation DataFrames are empty")

        # Validate required columns
        for name, df in [('train', self.train_df), ('val', self.val_df)]:
            if not df.empty:
                required_cols = [self.constants.FILENAME_COLUMN, self.constants.TARGET_COLUMN]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in {name} DataFrame: {missing_cols}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets based on stage.

        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        # Set seeds for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Create transforms
        train_transforms = self._create_training_transforms()
        val_transforms = self._create_validation_transforms()

        if stage in ('fit', None):
            # Create training and validation datasets
            if not self.train_df.empty:
                self.train_dataset = self._create_dataset(
                    self.train_df,
                    train_transforms,
                    "training"
                )

            if not self.val_df.empty:
                self.val_dataset = self._create_dataset(
                    self.val_df,
                    val_transforms,
                    "validation"
                )

        if stage in ('validate', None):
            # Create validation dataset if not already created
            if self.val_dataset is None and not self.val_df.empty:
                self.val_dataset = self._create_dataset(
                    self.val_df,
                    val_transforms,
                    "validation"
                )

        if stage in ('test', None):
            # Create test dataset if test data is available
            if self.test_df is not None:
                self.test_dataset = self._create_dataset(
                    self.test_df,
                    val_transforms,  # Use same transforms as validation
                    "test"
                )

        self.logger.info(f"Setup completed for stage: {stage}")

    def _create_dataset(
        self,
        dataframe: pd.DataFrame,
        transforms,
        dataset_type: str
    ) -> CustomImageDataset:
        """
        Create a CustomImageDataset instance.

        Args:
            dataframe: DataFrame with image data
            transforms: Transform pipeline to apply
            dataset_type: Type description for logging

        Returns:
            CustomImageDataset instance
        """
        dataset = CustomImageDataset(
            dataframe=dataframe,
            image_dir=self.image_dir,
            constants=self.constants,
            transform=transforms,
            color_mode=self.color_mode,
            validate_images=self.validate_images_on_init
        )

        # Log dataset statistics
        if len(dataset) > 0:
            class_dist = dataset.get_class_distribution()
            memory_info = dataset.get_memory_usage_estimate()

            self.logger.info(
                f"{dataset_type.capitalize()} dataset created: "
                f"{len(dataset)} samples, "
                f"classes: {class_dist}, "
                f"estimated memory: {memory_info['estimated_total_memory_mb']:.1f} MB"
            )
        else:
            self.logger.warning(f"{dataset_type.capitalize()} dataset is empty")

        return dataset

    def _create_training_transforms(self):
        """Create training transforms."""
        return self.transform_builder.build_training_transforms(
            enable_augmentation=True,
            augmentation_strength=getattr(self.config, 'augmentation_strength', 1.0),
            custom_preprocessing=self.custom_preprocessing_config if self.custom_preprocessing_config else None
        )

    def _create_validation_transforms(self):
        """Create validation/test transforms."""
        return self.transform_builder.build_validation_transforms(
            custom_preprocessing=self.custom_preprocessing_config if self.custom_preprocessing_config else None
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create training data loader.

        Returns:
            DataLoader for training data
        """
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        # Build dataloader kwargs based on num_workers
        loader_kwargs = {
            'batch_size': self.config.batch_size,
            'shuffle': True,
            'num_workers': self.config.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': False,
        }

        # Only set these parameters when using multiprocessing (num_workers > 0)
        if self.config.num_workers > 0:
            loader_kwargs['persistent_workers'] = self.persistent_workers
            loader_kwargs['prefetch_factor'] = self.prefetch_factor
            loader_kwargs['worker_init_fn'] = self._worker_init_fn

        return DataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """
        Create validation data loader.

        Returns:
            DataLoader for validation data
        """
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")

        # Build dataloader kwargs based on num_workers
        loader_kwargs = {
            'batch_size': self.config.batch_size,
            'shuffle': False,
            'num_workers': self.config.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': False,
        }

        # Only set these parameters when using multiprocessing (num_workers > 0)
        if self.config.num_workers > 0:
            loader_kwargs['persistent_workers'] = self.persistent_workers
            loader_kwargs['prefetch_factor'] = self.prefetch_factor
            loader_kwargs['worker_init_fn'] = self._worker_init_fn

        return DataLoader(self.val_dataset, **loader_kwargs)

    def test_dataloader(self) -> Optional[DataLoader]:
        """
        Create test data loader if test data available.

        Returns:
            DataLoader for test data, or None if no test data
        """
        if self.test_dataset is None:
            return None

        # Build dataloader kwargs based on num_workers
        loader_kwargs = {
            'batch_size': self.config.batch_size,
            'shuffle': False,
            'num_workers': self.config.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': False,
        }

        # Only set these parameters when using multiprocessing (num_workers > 0)
        if self.config.num_workers > 0:
            loader_kwargs['persistent_workers'] = self.persistent_workers
            loader_kwargs['prefetch_factor'] = self.prefetch_factor
            loader_kwargs['worker_init_fn'] = self._worker_init_fn

        return DataLoader(self.test_dataset, **loader_kwargs)



    def _worker_init_fn(self, worker_id: int) -> None:
        """
        Initialize worker processes with different random seeds.

        Args:
            worker_id: Worker process ID
        """
        # Set different seed for each worker to avoid duplicate augmentations
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed + worker_id)
        torch.manual_seed(worker_seed + worker_id)


   