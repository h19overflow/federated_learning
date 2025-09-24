"""
Entity classes for the federated pneumonia detection system.
Contains data structures and configuration classes.
"""

from .system_constants import SystemConstants
from .experiment_config import ExperimentConfig
from .custom_image_dataset import CustomImageDataset
from .resnet_with_custom_head import ResNetWithCustomHead

__all__ = [
    'SystemConstants',
    'ExperimentConfig',
    'CustomImageDataset',
    'ResNetWithCustomHead'
]