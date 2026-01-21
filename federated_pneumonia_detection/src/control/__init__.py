"""
Control layer for the federated pneumonia detection system.
Contains business logic and workflow orchestration classes.
"""

from federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet import (
    LitResNet,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.xray_data_module import (
    XRayDataModule,
)

__all__ = ["XRayDataModule", "LitResNet"]
