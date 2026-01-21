"""
Entity classes for the federated pneumonia detection system.
Contains data structures and configuration classes.

NOTE: SystemConstants and ExperimentConfig have been deprecated in favor of ConfigManager.
See federated_pneumonia_detection.config.config_manager for the new configuration approach.
"""

from .custom_image_dataset import CustomImageDataset
from .resnet_with_custom_head import ResNetWithCustomHead

__all__ = ["CustomImageDataset", "ResNetWithCustomHead"]
