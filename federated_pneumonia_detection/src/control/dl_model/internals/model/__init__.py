"""
Model utilities for PyTorch Lightning training.

This module provides Lightning modules, callbacks, optimizers, and collectors
for centralized and federated training.
"""

# Lightning Modules (surface level)
# Submodules (expose for convenience)
from . import callbacks, collectors, optimizers

# Focal Loss
from .losses import FocalLoss, FocalLossWithLabelSmoothing
from .lit_resnet_enhanced import LitResNetEnhanced

# Data Module
from ..data.xray_data_module import XRayDataModule

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
