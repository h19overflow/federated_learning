"""
Builder for X-ray transformation pipelines.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torchvision.transforms as transforms

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

from federated_pneumonia_detection.src.internals.loggers.logger import get_logger
from federated_pneumonia_detection.src.internals.transforms.base import XRayPreprocessor


class TransformBuilder:
    """Builder class for creating configurable transform pipelines."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        use_imagenet_norm: bool = True,
        augmentation_strength: float = 1.0,
    ):
        """Initialize transform builder."""
        self.img_size = img_size
        self.use_imagenet_norm = use_imagenet_norm
        self.augmentation_strength = augmentation_strength
        self.preprocessor = XRayPreprocessor()
        self.logger = get_logger(__name__)

    @classmethod
    def from_config(
        cls, config: Optional["ConfigManager"] = None
    ) -> "TransformBuilder":
        """Create TransformBuilder from a ConfigManager instance."""
        if config is None:
            from federated_pneumonia_detection.config.config_manager import (
                ConfigManager,
            )

            config = ConfigManager()

        return cls(
            img_size=tuple(config.get("system.img_size", [224, 224])),
            use_imagenet_norm=config.get("system.use_imagenet_norm", True),
            augmentation_strength=config.get("experiment.augmentation_strength", 1.0),
        )

    def build_training_transforms(
        self,
        enable_augmentation: bool = True,
        augmentation_strength: Optional[float] = None,
        custom_preprocessing: Optional[dict] = None,
    ) -> transforms.Compose:
        """Build transform pipeline for training data."""
        if augmentation_strength is None:
            augmentation_strength = self.augmentation_strength

        img_size = self.img_size
        transform_list = []

        if enable_augmentation:
            min_scale = max(0.7, 1.0 - augmentation_strength * 0.3)
            transform_list.extend(
                [
                    transforms.RandomResizedCrop(
                        img_size,
                        scale=(min_scale, 1.0),
                        ratio=(0.8, 1.2),
                    ),
                    transforms.RandomHorizontalFlip(p=0.5 * augmentation_strength),
                    transforms.RandomRotation(
                        degrees=15 * augmentation_strength,
                        fill=0,
                    ),
                    transforms.ColorJitter(
                        brightness=0.1 * augmentation_strength,
                        contrast=0.1 * augmentation_strength,
                        saturation=0.1 * augmentation_strength,
                        hue=0.05 * augmentation_strength,
                    ),
                ],
            )
        else:
            transform_list.extend(
                [transforms.Resize(img_size), transforms.CenterCrop(img_size)],
            )

        if custom_preprocessing:
            preprocess_fn = self.preprocessor.create_custom_preprocessing_pipeline(
                **custom_preprocessing,
            )
            transform_list.append(transforms.Lambda(preprocess_fn))

        transform_list.append(transforms.ToTensor())
        transform_list.extend(self._get_normalization_transforms())

        return transforms.Compose(transform_list)

    def build_validation_transforms(
        self,
        custom_preprocessing: Optional[dict] = None,
    ) -> transforms.Compose:
        """Build transform pipeline for validation/test data."""
        img_size = self.img_size
        transform_list = [transforms.Resize(img_size), transforms.CenterCrop(img_size)]

        if custom_preprocessing:
            preprocess_fn = self.preprocessor.create_custom_preprocessing_pipeline(
                **custom_preprocessing,
            )
            transform_list.append(transforms.Lambda(preprocess_fn))

        transform_list.append(transforms.ToTensor())
        transform_list.extend(self._get_normalization_transforms())

        return transforms.Compose(transform_list)

    def build_test_time_augmentation_transforms(
        self,
        num_augmentations: int = 5,
    ) -> list:
        """Build multiple transform pipelines for test-time augmentation."""
        transforms_list = []
        img_size = self.img_size

        for i in range(num_augmentations):
            rotation_angle = (i - num_augmentations // 2) * 5
            flip_prob = 1.0 if i % 2 == 0 else 0.0

            transform_list = [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.RandomRotation(degrees=(rotation_angle, rotation_angle)),
                transforms.RandomHorizontalFlip(p=flip_prob),
                transforms.ToTensor(),
            ]
            transform_list.extend(self._get_normalization_transforms())
            transforms_list.append(transforms.Compose(transform_list))

        return transforms_list

    def _get_normalization_transforms(self) -> list:
        """Get normalization transforms."""
        self.logger.info(f"use_imagenet_norm: {self.use_imagenet_norm}")
        if self.use_imagenet_norm:
            return [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        else:
            return [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    def create_custom_transform(
        self,
        resize_strategy: str = "resize_crop",
        augmentations: Optional[list] = None,
        preprocessing: Optional[dict] = None,
        normalization: str = "imagenet",
    ) -> transforms.Compose:
        """Create fully custom transform pipeline."""
        transform_list = []
        img_size = self.img_size

        if resize_strategy == "resize_crop":
            transform_list.extend(
                [transforms.Resize(img_size), transforms.CenterCrop(img_size)],
            )
        elif resize_strategy == "random_crop":
            transform_list.append(transforms.RandomResizedCrop(img_size))
        elif resize_strategy == "pad_resize":
            transform_list.append(
                transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
            )

        if augmentations:
            for aug in augmentations:
                if aug == "horizontal_flip":
                    transform_list.append(transforms.RandomHorizontalFlip())
                elif aug == "rotation":
                    transform_list.append(transforms.RandomRotation(15))
                elif aug == "color_jitter":
                    transform_list.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.05))
                elif aug == "gaussian_blur":
                    transform_list.append(transforms.GaussianBlur(kernel_size=3))

        if preprocessing:
            preprocess_fn = self.preprocessor.create_custom_preprocessing_pipeline(
                **preprocessing,
            )
            transform_list.append(transforms.Lambda(preprocess_fn))

        transform_list.append(transforms.ToTensor())

        if normalization == "imagenet":
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            )
        return transforms.Compose(transform_list)
