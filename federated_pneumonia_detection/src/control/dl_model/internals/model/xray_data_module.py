"""
PyTorch Lightning DataModule for X-ray image data management.
Orchestrates dataset creation, data loading, and batch management with comprehensive configuration.
"""

import logging
from typing import Optional, Union, Dict, Any, TYPE_CHECKING
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

from federated_pneumonia_detection.src.internals.image_transforms import TransformBuilder
from federated_pneumonia_detection.src.control.dl_model.internals.model.data_module_utils import (
    validate_inputs,
    create_dataset,
    create_training_transforms,
    create_validation_transforms,
    build_dataloader_kwargs,
)


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
        super().__init__()

        if config is None:
            from federated_pneumonia_detection.config.config_manager import ConfigManager
            config = ConfigManager()

        self.config = config
        self.save_hyperparameters(ignore=['train_df', 'val_df', 'test_df', 'config'])

        self.train_df = train_df.copy() if not train_df.empty else train_df
        self.val_df = val_df.copy() if not val_df.empty else val_df
        self.test_df = test_df.copy() if test_df is not None and not test_df.empty else None
        self.image_dir = Path(image_dir) if image_dir else Path('.')

        self.color_mode = (color_mode or self.config.get('experiment.color_mode', 'RGB')).upper()
        num_workers = self.config.get('experiment.num_workers', 4)
        self.validate_images_on_init = validate_images_on_init if validate_images_on_init is not None else self.config.get('experiment.validate_images_on_init', True)

        self.filename_column = self.config.get('columns.filename', 'filename')
        self.target_column = self.config.get('columns.target', 'Target')

        self.pin_memory = (pin_memory if pin_memory is not None else self.config.get('experiment.pin_memory', True)) and torch.cuda.is_available()
        self.persistent_workers = (persistent_workers if persistent_workers is not None else self.config.get('experiment.persistent_workers', False)) and num_workers > 0
        self.prefetch_factor = prefetch_factor if prefetch_factor is not None else (self.config.get('experiment.prefetch_factor', 2) if num_workers > 0 else 2)

        self.custom_preprocessing_config = custom_preprocessing_config or {}

        self.logger = logging.getLogger(__name__)
        self.transform_builder = TransformBuilder(config=self.config)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        validate_inputs(
            image_dir=self.image_dir,
            color_mode=self.color_mode,
            train_df=self.train_df,
            val_df=self.val_df,
            filename_column=self.filename_column,
            target_column=self.target_column,
        )

        self.logger.info(f"DataModule initialized - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df) if test_df is not None else 0}")

    def setup(self, stage: Optional[str] = None) -> None:
        seed = self.config.get('experiment.seed', 42)
        torch.manual_seed(seed)
        import numpy as np
        np.random.seed(seed)

        train_transforms = create_training_transforms(
            self.transform_builder,
            self.custom_preprocessing_config if self.custom_preprocessing_config else None
        )
        val_transforms = create_validation_transforms(
            self.transform_builder,
            self.custom_preprocessing_config if self.custom_preprocessing_config else None
        )

        if stage in ('fit', None):
            if not self.train_df.empty:
                self.train_dataset = create_dataset(
                    dataframe=self.train_df,
                    image_dir=self.image_dir,
                    config=self.config,
                    transforms=train_transforms,
                    color_mode=self.color_mode,
                    validate_images=self.validate_images_on_init,
                    dataset_type="training"
                )

            if not self.val_df.empty:
                self.val_dataset = create_dataset(
                    dataframe=self.val_df,
                    image_dir=self.image_dir,
                    config=self.config,
                    transforms=val_transforms,
                    color_mode=self.color_mode,
                    validate_images=self.validate_images_on_init,
                    dataset_type="validation"
                )

        if stage in ('validate', None):
            if self.val_dataset is None and not self.val_df.empty:
                self.val_dataset = create_dataset(
                    dataframe=self.val_df,
                    image_dir=self.image_dir,
                    config=self.config,
                    transforms=val_transforms,
                    color_mode=self.color_mode,
                    validate_images=self.validate_images_on_init,
                    dataset_type="validation"
                )

        if stage in ('test', None):
            if self.test_df is not None:
                self.test_dataset = create_dataset(
                    dataframe=self.test_df,
                    image_dir=self.image_dir,
                    config=self.config,
                    transforms=val_transforms,
                    color_mode=self.color_mode,
                    validate_images=self.validate_images_on_init,
                    dataset_type="test"
                )

        self.logger.info(f"Setup completed for stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        loader_kwargs = build_dataloader_kwargs(
            config=self.config,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

        return DataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")

        loader_kwargs = build_dataloader_kwargs(
            config=self.config,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

        return DataLoader(self.val_dataset, **loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            return None

        loader_kwargs = build_dataloader_kwargs(
            config=self.config,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

        return DataLoader(self.test_dataset, **loader_kwargs)
