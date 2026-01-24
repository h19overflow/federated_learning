"""
Loss functions for pneumonia detection models.

This module provides specialized loss functions for handling class imbalance
in medical imaging tasks, including Focal Loss and variants with label smoothing.
"""

from .focal_loss import FocalLoss, FocalLossWithLabelSmoothing

__all__ = [
    "FocalLoss",
    "FocalLossWithLabelSmoothing",
]
