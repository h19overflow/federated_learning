"""
Model utilities for PyTorch Lightning training.

This module provides Lightning modules, callbacks, optimizers, and collectors
for centralized and federated training.
"""

# Lightning Modules (surface level)
# Submodules (expose for convenience)
# Data Module
from ..data.xray_data_module import XRayDataModule
from . import callbacks, collectors, optimizers
from .lit_resnet_enhanced import LitResNetEnhanced

# Focal Loss
from .losses import FocalLoss, FocalLossWithLabelSmoothing

__all__ = [
    # Lightning Modules
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
