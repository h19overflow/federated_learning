"""
Loss function factory and calculation logic for LitResNetEnhanced.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from federated_pneumonia_detection.src.control.dl_model.internals.model.losses import (
    FocalLoss,
    FocalLossWithLabelSmoothing,
)


class LossFactory:
    """
    Factory for creating and managing loss functions.
    Centralizes loss initialization and calculation logic.
    """

    @staticmethod
    def create_loss_function(
        class_weights_tensor: Optional[torch.Tensor] = None,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        logger: Optional[logging.Logger] = None,
    ) -> nn.Module:
        """
        Setup loss function with enhanced options.

        Args:
            class_weights_tensor: Optional class weights for loss
            use_focal_loss: Whether to use Focal Loss
            focal_alpha: Alpha parameter for Focal Loss
            focal_gamma: Gamma parameter for Focal Loss
            label_smoothing: Label smoothing factor (0 to disable)
            logger: Optional logger for info messages

        Returns:
            Initialized loss function
        """
        pos_weight = None
        if class_weights_tensor is not None:
            pos_weight = class_weights_tensor[1] / (class_weights_tensor[0] + 1e-8)
            if logger:
                logger.info(f"Using positive class weight: {pos_weight}")

        if use_focal_loss:
            if label_smoothing > 0:
                if logger:
                    logger.info(
                        f"Using FocalLoss with label smoothing ({label_smoothing})"
                    )
                return FocalLossWithLabelSmoothing(
                    alpha=focal_alpha,
                    gamma=focal_gamma,
                    smoothing=label_smoothing,
                    pos_weight=pos_weight,
                )
            else:
                if logger:
                    logger.info("Using FocalLoss")
                return FocalLoss(
                    alpha=focal_alpha,
                    gamma=focal_gamma,
                    pos_weight=pos_weight,
                )
        else:
            if logger:
                logger.info("Using BCEWithLogitsLoss")
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    @staticmethod
    def calculate_loss(
        loss_fn: nn.Module,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate loss based on task type.

        Args:
            loss_fn: The loss function to use
            logits: Model output logits
            targets: Ground truth targets

        Returns:
            Calculated loss tensor
        """
        targets = (
            targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()
        )
        return loss_fn(logits, targets)
