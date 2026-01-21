"""
Image transformation utilities for X-ray preprocessing and augmentation.
Provides configurable transform pipelines and custom preprocessing functions.
"""

import logging
from typing import Optional, Callable, TYPE_CHECKING
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

from federated_pneumonia_detection.src.internals.loggers.logger import get_logger


class XRayPreprocessor:
    """
    Custom preprocessing utilities for X-ray images.

    Provides contrast enhancement and other X-ray specific preprocessing.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize preprocessor.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or get_logger(__name__)

    @staticmethod
    def contrast_stretch_percentile(
        image: Image.Image,
        lower_percentile: float = 5.0,
        upper_percentile: float = 95.0,
        eps: float = 1e-8,
    ) -> Image.Image:
        """
        Apply percentile-based contrast stretching to enhance X-ray visibility.

        Args:
            image: Input PIL image
            lower_percentile: Lower percentile for contrast stretching
            upper_percentile: Upper percentile for contrast stretching
            eps: Small constant to prevent division by zero

        Returns:
            Contrast-enhanced PIL image
        """
        # Convert to numpy array and normalize to [0,1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Calculate percentiles
        p_lower, p_upper = np.percentile(
            img_array, (lower_percentile, upper_percentile)
        )

        # Apply contrast stretching
        img_array = np.clip((img_array - p_lower) / (p_upper - p_lower + eps), 0, 1)

        # Convert back to PIL image
        mode = "RGB" if img_array.ndim == 3 else "L"
        return Image.fromarray((img_array * 255).astype(np.uint8), mode=mode)

    @staticmethod
    def adaptive_histogram_equalization(
        image: Image.Image, clip_limit: float = 2.0
    ) -> Image.Image:
        """
        Apply adaptive histogram equalization (CLAHE) for better contrast.
        NOTE: This implementation currently returns the original image as cv2 is not available.

        Args:
            image: Input PIL image
            clip_limit: Contrast limiting parameter

        Returns:
            Original image (CLAHE currently disabled)
        """
        logging.warning("cv2 not installed, skipping adaptive histogram equalization")
        return image

    @staticmethod
    def edge_enhancement(image: Image.Image, strength: float = 1.0) -> Image.Image:
        """
        Apply edge enhancement to highlight anatomical structures.

        Args:
            image: Input PIL image
            strength: Enhancement strength (0.0 to 2.0)

        Returns:
            Edge-enhanced PIL image
        """
        from PIL import ImageFilter

        # Apply unsharp mask filter
        enhanced = image.filter(
            ImageFilter.UnsharpMask(radius=2, percent=int(strength * 150), threshold=3)
        )

        return enhanced

    def create_custom_preprocessing_pipeline(
        self,
        contrast_stretch: bool = True,
        adaptive_hist: bool = False,
        edge_enhance: bool = False,
        **kwargs,
    ) -> Callable[[Image.Image], Image.Image]:
        """
        Create custom preprocessing pipeline for X-ray images.

        Args:
            contrast_stretch: Whether to apply percentile contrast stretching
            adaptive_hist: Whether to apply adaptive histogram equalization
            edge_enhance: Whether to apply edge enhancement
            **kwargs: Additional parameters for preprocessing functions

        Returns:
            Preprocessing function that can be used in transforms
        """

        def preprocess_fn(image: Image.Image) -> Image.Image:
            try:
                result = image

                if contrast_stretch:
                    result = self.contrast_stretch_percentile(
                        result,
                        lower_percentile=kwargs.get("lower_percentile", 5.0),
                        upper_percentile=kwargs.get("upper_percentile", 95.0),
                    )

                if adaptive_hist:
                    result = self.adaptive_histogram_equalization(
                        result, clip_limit=kwargs.get("clip_limit", 2.0)
                    )

                if edge_enhance:
                    result = self.edge_enhancement(
                        result, strength=kwargs.get("edge_strength", 1.0)
                    )

                return result

            except Exception as e:
                self.logger.warning(
                    f"Preprocessing failed, returning original image: {e}"
                )
                return image

        return preprocess_fn


class TransformBuilder:
    """
    Builder class for creating configurable transform pipelines.

    Provides flexible and extensible transform pipeline creation.
    """

    def __init__(self, config: Optional["ConfigManager"] = None):
        """
        Initialize transform builder.

        Args:
            config: ConfigManager for configuration
        """
        if config is None:
            from federated_pneumonia_detection.config.config_manager import (
                ConfigManager,
            )

            config = ConfigManager()

        self.config = config
        self.preprocessor = XRayPreprocessor()
        self.logger = get_logger(__name__)

    def build_training_transforms(
        self,
        enable_augmentation: bool = True,
        augmentation_strength: Optional[float] = None,
        custom_preprocessing: Optional[dict] = None,
    ) -> transforms.Compose:
        """
        Build transform pipeline for training data.

        Args:
            enable_augmentation: Whether to include data augmentation
            augmentation_strength: Strength of augmentation (0.0 to 2.0)
            custom_preprocessing: Dict with custom preprocessing options

        Returns:
            Composed transform pipeline
        """
        if augmentation_strength is None:
            augmentation_strength = self.config.get(
                "experiment.augmentation_strength", 1.0
            )

        img_size = tuple(self.config.get("system.img_size", [224, 224]))

        transform_list = []

        # Resize and augmentation
        if enable_augmentation:
            # Random resized crop with configurable scale
            min_scale = max(0.7, 1.0 - augmentation_strength * 0.3)
            transform_list.extend(
                [
                    transforms.RandomResizedCrop(
                        img_size, scale=(min_scale, 1.0), ratio=(0.8, 1.2)
                    ),
                    transforms.RandomHorizontalFlip(p=0.5 * augmentation_strength),
                    transforms.RandomRotation(
                        degrees=15 * augmentation_strength, fill=0
                    ),
                    transforms.ColorJitter(
                        brightness=0.1 * augmentation_strength,
                        contrast=0.1 * augmentation_strength,
                        saturation=0.1 * augmentation_strength,
                        hue=0.05 * augmentation_strength,
                    ),
                ]
            )
        else:
            transform_list.extend(
                [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
            )

        # Custom preprocessing if specified
        if custom_preprocessing:
            preprocess_fn = self.preprocessor.create_custom_preprocessing_pipeline(
                **custom_preprocessing
            )
            transform_list.append(transforms.Lambda(preprocess_fn))

        # Convert to tensor and normalize
        transform_list.append(transforms.ToTensor())
        transform_list.extend(self._get_normalization_transforms())

        return transforms.Compose(transform_list)

    def build_validation_transforms(
        self, custom_preprocessing: Optional[dict] = None
    ) -> transforms.Compose:
        """
        Build transform pipeline for validation/test data.

        Args:
            custom_preprocessing: Dict with custom preprocessing options

        Returns:
            Composed transform pipeline
        """
        transform_list = []
        img_size = tuple(self.config.get("system.img_size", [224, 224]))

        # Simple resize and crop for validation
        transform_list.extend(
            [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
        )

        # Custom preprocessing if specified
        if custom_preprocessing:
            preprocess_fn = self.preprocessor.create_custom_preprocessing_pipeline(
                **custom_preprocessing
            )
            transform_list.append(transforms.Lambda(preprocess_fn))

        # Convert to tensor and normalize
        transform_list.append(transforms.ToTensor())
        transform_list.extend(self._get_normalization_transforms())

        return transforms.Compose(transform_list)

    def build_test_time_augmentation_transforms(
        self, num_augmentations: int = 5
    ) -> list:
        """
        Build multiple transform pipelines for test-time augmentation.

        Args:
            num_augmentations: Number of augmented versions to create

        Returns:
            List of transform pipelines
        """
        transforms_list = []
        img_size = tuple(self.config.get("system.img_size", [224, 224]))

        for i in range(num_augmentations):
            transform_list = []

            # Vary the augmentation for each version
            rotation_angle = (
                i - num_augmentations // 2
            ) * 5  # -10, -5, 0, 5, 10 degrees
            flip_prob = 1.0 if i % 2 == 0 else 0.0

            transform_list.extend(
                [
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.RandomRotation(degrees=(rotation_angle, rotation_angle)),
                    transforms.RandomHorizontalFlip(p=flip_prob),
                    transforms.ToTensor(),
                ]
            )

            transform_list.extend(self._get_normalization_transforms())
            transforms_list.append(transforms.Compose(transform_list))

        return transforms_list

    def _get_normalization_transforms(self) -> list:
        """
        Get normalization transforms based on configuration.

        Returns:
            List of normalization transforms
        """
        # Check if ImageNet normalization should be used
        use_imagenet_norm = self.config.get("system.use_imagenet_norm", True)
        self.logger.info(f"use_imagenet_norm: {use_imagenet_norm}")

        if use_imagenet_norm:
            # ImageNet normalization values
            return [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        else:
            # Simple normalization to [-1, 1]
            return [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    def create_custom_transform(
        self,
        resize_strategy: str = "resize_crop",
        augmentations: Optional[list] = None,
        preprocessing: Optional[dict] = None,
        normalization: str = "imagenet",
    ) -> transforms.Compose:
        """
        Create fully custom transform pipeline.

        Args:
            resize_strategy: 'resize_crop', 'random_crop', or 'pad_resize'
            augmentations: List of augmentation names to apply
            preprocessing: Custom preprocessing parameters
            normalization: 'imagenet', 'zero_one', or 'minus_one_one'

        Returns:
            Custom transform pipeline
        """
        transform_list = []
        img_size = tuple(self.config.get("system.img_size", [224, 224]))

        # Resize strategy
        if resize_strategy == "resize_crop":
            transform_list.extend(
                [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
            )
        elif resize_strategy == "random_crop":
            transform_list.append(transforms.RandomResizedCrop(img_size))
        elif resize_strategy == "pad_resize":
            transform_list.append(
                transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR)
            )

        # Augmentations
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

        # Custom preprocessing
        if preprocessing:
            preprocess_fn = self.preprocessor.create_custom_preprocessing_pipeline(
                **preprocessing
            )
            transform_list.append(transforms.Lambda(preprocess_fn))

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalization
        if normalization == "imagenet":
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )
        return transforms.Compose(transform_list)


# Convenience functions for backward compatibility and ease of use
def get_transforms(
    config: Optional["ConfigManager"],
    is_training: bool = True,
    use_custom_preprocessing: bool = False,
    augmentation_strength: Optional[float] = None,
    **kwargs,
) -> transforms.Compose:
    """
    Get transform pipeline using the builder pattern.

    Args:
        config: ConfigManager instance
        is_training: Whether transforms are for training
        use_custom_preprocessing: Whether to apply custom X-ray preprocessing
        augmentation_strength: Strength of data augmentation
        **kwargs: Additional preprocessing parameters

    Returns:
        Transform pipeline
    """
    builder = TransformBuilder(config)

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
            custom_preprocessing=preprocessing_config
        )


def create_preprocessing_function(
    config: Optional["ConfigManager"], contrast_stretch: bool = True, **kwargs
) -> Callable[[Image.Image], Image.Image]:
    """
    Create standalone preprocessing function for X-ray images.

    Args:
        config: ConfigManager instance
        contrast_stretch: Whether to apply contrast stretching
        **kwargs: Additional preprocessing parameters

    Returns:
        Preprocessing function
    """
    preprocessor = XRayPreprocessor()
    return preprocessor.create_custom_preprocessing_pipeline(
        contrast_stretch=contrast_stretch, **kwargs
    )
