"""
Image transformation utilities for X-ray preprocessing and augmentation.
Provides configurable transform pipelines and custom preprocessing functions.
This module acts as a facade for the underlying transform components.
"""

from typing import TYPE_CHECKING, Callable, Optional

import torchvision.transforms as transforms
from PIL import Image

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

from federated_pneumonia_detection.src.internals.transforms.base import XRayPreprocessor
from federated_pneumonia_detection.src.internals.transforms.builder import (
    TransformBuilder,
)


def get_transforms(
    config: Optional["ConfigManager"],
    is_training: bool = True,
    use_custom_preprocessing: bool = False,
    augmentation_strength: Optional[float] = None,
    **kwargs,
) -> transforms.Compose:
    """Get transform pipeline using the builder pattern."""
    builder = TransformBuilder.from_config(config)

    preprocessing_config = None
    if use_custom_preprocessing:
        preprocessing_config = {
            "contrast_stretch": True,
            "adaptive_hist": kwargs.get("adaptive_hist", False),
            "edge_enhance": kwargs.get("edge_enhance", False),
            **{
                k: v
                for k, v in kwargs.items()
                if k.startswith(("lower_", "upper_", "clip_", "edge_"))
            },
        }

    if is_training:
        return builder.build_training_transforms(
            enable_augmentation=True,
            augmentation_strength=augmentation_strength,
            custom_preprocessing=preprocessing_config,
        )
    else:
        return builder.build_validation_transforms(
            custom_preprocessing=preprocessing_config,
        )


def create_preprocessing_function(
    config: Optional["ConfigManager"],
    contrast_stretch: bool = True,
    **kwargs,
) -> Callable[[Image.Image], Image.Image]:
    """Create standalone preprocessing function for X-ray images."""
    preprocessor = XRayPreprocessor()
    return preprocessor.create_custom_preprocessing_pipeline(
        contrast_stretch=contrast_stretch,
        **kwargs,
    )


# Re-export classes for backward compatibility
__all__ = [
    "get_transforms",
    "create_preprocessing_function",
    "XRayPreprocessor",
    "TransformBuilder",
]
