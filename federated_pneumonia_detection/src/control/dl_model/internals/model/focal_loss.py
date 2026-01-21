"""
Focal Loss implementation for handling class imbalance in medical imaging.
Focuses learning on hard examples by down-weighting well-classified examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.

    Focal Loss reduces the relative loss for well-classified examples,
    focusing training on hard misclassified examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        p_t = p if y=1, else (1-p)
        alpha_t = alpha if y=1, else (1-alpha)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for the positive class (default 0.25).
                   Set to -1 to disable alpha weighting.
            gamma: Focusing parameter. Higher values focus more on hard examples.
                   gamma=0 reduces to standard cross-entropy.
            pos_weight: Optional weight for positive class (from class imbalance).
            reduction: Reduction method - 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Raw logits from model (N, 1) or (N,)
            targets: Binary targets (N, 1) or (N,)

        Returns:
            Focal loss value
        """
        # Ensure proper shapes
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        targets = targets.float()

        # Compute binary cross-entropy with logits (more stable)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute probabilities
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting if enabled
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        # Apply pos_weight if provided (additional class imbalance handling)
        if self.pos_weight is not None:
            pos_weight_t = self.pos_weight * targets + 1 * (1 - targets)
            focal_weight = focal_weight * pos_weight_t

        # Compute focal loss
        focal_loss = focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss with label smoothing for better generalization.

    Combines focal loss focusing with label smoothing regularization.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss with Label Smoothing.

        Args:
            alpha: Weighting factor for the positive class.
            gamma: Focusing parameter.
            smoothing: Label smoothing factor (0.0 to 0.5).
            pos_weight: Optional weight for positive class.
            reduction: Reduction method.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.reduction = reduction

        if not 0.0 <= smoothing < 0.5:
            raise ValueError("Smoothing must be in [0, 0.5)")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss with label smoothing.

        Args:
            inputs: Raw logits from model
            targets: Binary targets

        Returns:
            Focal loss with label smoothing
        """
        # Ensure proper shapes
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        # Apply label smoothing
        targets = targets.float()
        targets_smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        # Compute binary cross-entropy with smoothed labels
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets_smoothed, reduction="none"
        )

        # Use original (non-smoothed) targets for focal weighting
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        # Apply pos_weight if provided
        if self.pos_weight is not None:
            pos_weight_t = self.pos_weight * targets + 1 * (1 - targets)
            focal_weight = focal_weight * pos_weight_t

        # Compute focal loss
        focal_loss = focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
