"""Dataset creation utilities for XRayDataModule."""

import logging
from typing import Optional, Dict, Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

from federated_pneumonia_detection.src.entities.custom_image_dataset import CustomImageDataset
from federated_pneumonia_detection.src.utils.image_transforms import TransformBuilder

logger = logging.getLogger(__name__)


def create_dataset(
    dataframe: pd.DataFrame,
    image_dir: 'Path',
    config: 'ConfigManager',
    transforms,
    color_mode: str,
    validate_images: bool,
    dataset_type: str,
) -> CustomImageDataset:
    """
    Create a CustomImageDataset instance with logging.

    Args:
        dataframe: DataFrame with image data
        image_dir: Directory containing images
        config: ConfigManager instance
        transforms: Transform pipeline to apply
        color_mode: Color mode ('RGB' or 'L')
        validate_images: Whether to validate images
        dataset_type: Type description for logging

    Returns:
        CustomImageDataset instance
    """
    dataset = CustomImageDataset(
        dataframe=dataframe,
        image_dir=image_dir,
        config=config,
        transform=transforms,
        color_mode=color_mode,
        validate_images=validate_images
    )

    # Log dataset statistics
    if len(dataset) > 0:
        class_dist = dataset.get_class_distribution()
        memory_info = dataset.get_memory_usage_estimate()

        logger.info(
            f"{dataset_type.capitalize()} dataset created: "
            f"{len(dataset)} samples, "
            f"classes: {class_dist}, "
            f"estimated memory: {memory_info['estimated_total_memory_mb']:.1f} MB"
        )
    else:
        logger.warning(f"{dataset_type.capitalize()} dataset is empty")

    return dataset


def create_training_transforms(
    transform_builder: TransformBuilder,
    custom_preprocessing_config: Optional[Dict[str, Any]] = None
):
    """
    Create training transforms with augmentation.

    Args:
        transform_builder: TransformBuilder instance
        custom_preprocessing_config: Custom preprocessing parameters

    Returns:
        Transform pipeline for training
    """
    return transform_builder.build_training_transforms(
        enable_augmentation=True,
        augmentation_strength=getattr(transform_builder.config, 'augmentation_strength', 1.0),
        custom_preprocessing=custom_preprocessing_config if custom_preprocessing_config else None
    )


def create_validation_transforms(
    transform_builder: TransformBuilder,
    custom_preprocessing_config: Optional[Dict[str, Any]] = None
):
    """
    Create validation/test transforms without augmentation.

    Args:
        transform_builder: TransformBuilder instance
        custom_preprocessing_config: Custom preprocessing parameters

    Returns:
        Transform pipeline for validation/test
    """
    return transform_builder.build_validation_transforms(
        custom_preprocessing=custom_preprocessing_config if custom_preprocessing_config else None
    )
