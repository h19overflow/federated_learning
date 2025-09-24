"""
Control layer for the federated pneumonia detection system.
Contains business logic and workflow orchestration classes.
"""

from .xray_data_module import XRayDataModule
from .lit_resnet import LitResNet

__all__ = [
    'XRayDataModule',
    'LitResNet'
]