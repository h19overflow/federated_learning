"""
PyTorch Lightning DataModule for X-ray image data management.
Orchestrates dataset creation, data loading, and batch management with comprehensive configuration.
"""

import logging
from typing import Optional, Union, Dict, Any, TYPE_CHECKING
from pathlib import Path

import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

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
        config: Optional['ConfigManager'] = None,
        image_dir: Union[str, Path] = None,
        test_df: Optional[pd.DataFrame] = None,
        color_mode: Optional[str] = None,
        pin_memory: Optional[bool] = None,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
        validate_images_on_init: Optional[bool] = None,
        custom_preprocessing_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the XRayDataModule.

        Args:
            train_df: DataFrame containing training data
            val_df: DataFrame containing validation data
            config: ConfigManager for configuration
            image_dir: Directory containing image files
            test_df: Optional DataFrame for test data
            color_mode: 'RGB' or 'L' for color mode (uses config default if None)
            pin_memory: Whether to use pinned memory (uses config if None)
            persistent_workers: Whether to keep workers alive (uses config if None)
            prefetch_factor: Number of batches to prefetch (uses config if None)
            validate_images_on_init: Whether to validate images (uses config if None)
            custom_preprocessing_config: Custom preprocessing parameters
        """
        super().__init__()

        if config is None:
            from federated_pneumonia_detection.config.config_manager import ConfigManager
            config = ConfigManager()

        self.config = config
        self.save_hyperparameters(ignore=['train_df', 'val_df', 'test_df', 'config'])

        # Store dataframes
        self.train_df = train_df.copy() if not train_df.empty else train_df
        self.val_df = val_df.copy() if not val_df.empty else val_df
        self.test_df = test_df.copy() if test_df is not None and not test_df.empty else None
        self.image_dir = Path(image_dir) if image_dir else Path('.')
        
        # Get configuration values with defaults from config
        self.color_mode = (color_mode or self.config.get('experiment.color_mode', 'RGB')).upper()
        num_workers = self.config.get('experiment.num_workers', 4)
        self.validate_images_on_init = validate_images_on_init if validate_images_on_init is not None else self.config.get('experiment.validate_images_on_init', True)
        
        # Column names from config
        self.filename_column = self.config.get('columns.filename', 'filename')
        self.target_column = self.config.get('columns.target', 'Target')

        # DataLoader configuration
        self.pin_memory = (pin_memory if pin_memory is not None else self.config.get('experiment.pin_memory', True)) and torch.cuda.is_available()
        self.persistent_workers = (persistent_workers if persistent_workers is not None else self.config.get('experiment.persistent_workers', False)) and num_workers > 0
        self.prefetch_factor = prefetch_factor if prefetch_factor is not None else (self.config.get('experiment.prefetch_factor', 2) if num_workers > 0 else 2)

        # Custom preprocessing
        self.custom_preprocessing_config = custom_preprocessing_config or {}

        # Initialize components
        self.logger = logging.getLogger(__name__)
        self.transform_builder = TransformBuilder(config=self.config)

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
                required_cols = [self.filename_column, self.target_column]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in {name} DataFrame: {missing_cols}")

    def setup(self, stage: Optional[str] = None, ) -> None:
        """
        Set up datasets based on stage.

        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        # Set seeds for reproducibility
        seed = self.config.get('experiment.seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

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
            config=self.config,
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
            'batch_size': self.config.get('experiment.batch_size', 32),
            'shuffle': True,
            'num_workers': self.config.get('experiment.num_workers', 4),
            'pin_memory': self.pin_memory,
            'drop_last': False,
        }

        # Only set these parameters when using multiprocessing (num_workers > 0)
        if self.config.get('experiment.num_workers', 4) > 0:
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
            'batch_size': self.config.get('experiment.batch_size', 32),
            'shuffle': False,
            'num_workers': self.config.get('experiment.num_workers', 4),
            'pin_memory': self.pin_memory,
            'drop_last': False,
        }

        # Only set these parameters when using multiprocessing (num_workers > 0)
        if self.config.get('experiment.num_workers', 4) > 0:
            loader_kwargs['persistent_workers'] = self.persistent_workers
            loader_kwargs['prefetch_factor'] = self.prefetch_factor
            loader_kwargs['worker_init_fn'] = self._worker_init_fn

        return DataLoader(self.val_dataset, **loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        """
        Create test data loader if test data available.

        Returns:
            DataLoader for test data, or None if no test data
        """
        if self.test_dataset is None:
            return None

        # Build dataloader kwargs based on num_workers
        loader_kwargs = {
            'batch_size': self.config.get('experiment.batch_size', 32),
            'shuffle': False,
            'num_workers': self.config.get('experiment.num_workers', 4),
            'pin_memory': self.pin_memory,
            'drop_last': False,
        }

        # Only set these parameters when using multiprocessing (num_workers > 0)
        if self.config.get('experiment.num_workers', 4) > 0:
            loader_kwargs['persistent_workers'] = self.persistent_workers
            loader_kwargs['prefetch_factor'] = self.prefetch_factor
            loader_kwargs['worker_init_fn'] = self._worker_init_fn

        return DataLoader(self.test_dataset, **loader_kwargs)



    def _worker_init_fn(self, worker_id: int) -> DataLoader:
        """
        Initialize worker processes with different random seeds.

        Args:
            worker_id: Worker process ID
        """
        # Set different seed for each worker to avoid duplicate augmentations
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed + worker_id)
        torch.manual_seed(worker_seed + worker_id)
