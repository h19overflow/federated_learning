"""
Control layer for the federated pneumonia detection system.
Contains business logic and workflow orchestration classes.
"""

from federated_pneumonia_detection.src.control.dl_model.utils.model.xray_data_module import XRayDataModule
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet import LitResNet

__all__ = [
    'XRayDataModule',
    'LitResNet'
]