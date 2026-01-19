"""
Model utilities for PyTorch Lightning training.

This module provides Lightning modules, callbacks, optimizers, and collectors
for centralized and federated training.
"""

# Lightning Modules (surface level)
from .lit_resnet import LitResNet
from .lit_resnet_enhanced import LitResNetEnhanced

# Data Module
from .xray_data_module import XRayDataModule

# Focal Loss
from .focal_loss import FocalLoss, FocalLossWithLabelSmoothing

# Submodules (expose for convenience)
from . import optimizers
from . import callbacks
from . import collectors

__all__ = [
    # Lightning Modules
    "LitResNet",
    "LitResNetEnhanced",
    # Data
    "XRayDataModule",
    # Losses
    "FocalLoss",
    "FocalLossWithLabelSmoothing",
    # Submodules
    "optimizers",
    "callbacks",
    "collectors",
]
