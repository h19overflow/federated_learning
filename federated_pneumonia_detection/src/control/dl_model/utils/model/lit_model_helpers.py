"""
Helper functions for PyTorch Lightning model modules.
Contains utility functions for model configuration, metrics setup, and summarization.
"""

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torchmetrics

from federated_pneumonia_detection.src.control.dl_model.utils.model.focal_loss import (
    FocalLoss,
    FocalLossWithLabelSmoothing,
)

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager


def validate_config(config: "ConfigManager", logger: logging.Logger) -> None:
    """
    Validate configuration parameters.

    Args:
        config: ConfigManager instance
        logger: Logger instance for messages

    Raises:
        ValueError: If configuration is invalid
    """
    lr = config.get("experiment.learning_rate", 0)
    if lr <= 0:
        raise ValueError("Learning rate must be positive")

    wd = config.get("experiment.weight_decay", -1)
    if wd < 0:
        raise ValueError("Weight decay must be non-negative")


def setup_metrics(
    num_classes: int,
    train: bool = True,
    validation: bool = True,
    test: bool = True
) -> Dict[str, torchmetrics.Metric]:
    """
    Initialize torchmetrics for tracking performance.

    Args:
        num_classes: Number of output classes
        train: Whether to setup training metrics
        validation: Whether to setup validation metrics
        test: Whether to setup test metrics

    Returns:
        Dictionary of metric names to metric instances
    """
    metrics = {}
    num_classes_metric = 2 if num_classes == 1 else num_classes
    task_type = "binary" if num_classes == 1 else "multiclass"

    if train:
        metrics["train_accuracy"] = torchmetrics.Accuracy(task=task_type, num_classes=num_classes_metric)
        metrics["train_f1"] = torchmetrics.F1Score(task=task_type, num_classes=num_classes_metric)

    if validation:
        metrics["val_accuracy"] = torchmetrics.Accuracy(task=task_type, num_classes=num_classes_metric)
        metrics["val_precision"] = torchmetrics.Precision(task=task_type, num_classes=num_classes_metric)
        metrics["val_recall"] = torchmetrics.Recall(task=task_type, num_classes=num_classes_metric)
        metrics["val_f1"] = torchmetrics.F1Score(task=task_type, num_classes=num_classes_metric)
        metrics["val_auroc"] = torchmetrics.AUROC(task=task_type, num_classes=num_classes_metric)
        metrics["val_confusion"] = torchmetrics.ConfusionMatrix(task=task_type, num_classes=num_classes_metric)

    if test:
        metrics["test_accuracy"] = torchmetrics.Accuracy(task=task_type, num_classes=num_classes_metric)
        metrics["test_precision"] = torchmetrics.Precision(task=task_type, num_classes=num_classes_metric)
        metrics["test_recall"] = torchmetrics.Recall(task=task_type, num_classes=num_classes_metric)
        metrics["test_f1"] = torchmetrics.F1Score(task=task_type, num_classes=num_classes_metric)
        metrics["test_auroc"] = torchmetrics.AUROC(task=task_type, num_classes=num_classes_metric)

    return metrics


def setup_loss_function(
    use_focal_loss: bool,
    focal_alpha: float,
    focal_gamma: float,
    label_smoothing: float,
    class_weights_tensor: Optional[torch.Tensor] = None,
    logger: Optional[logging.Logger] = None
) -> nn.Module:
    """
    Setup loss function with enhanced options.

    Args:
        use_focal_loss: Whether to use Focal Loss
        focal_alpha: Alpha parameter for Focal Loss
        focal_gamma: Gamma parameter for Focal Loss
        label_smoothing: Label smoothing factor (0 to disable)
        class_weights_tensor: Optional class weights for loss
        logger: Optional logger instance for messages

    Returns:
        Configured loss function
    """
    pos_weight = None
    if class_weights_tensor is not None:
        pos_weight = class_weights_tensor[1] / (class_weights_tensor[0] + 1e-8)
        if logger:
            logger.info(f"Using positive class weight: {pos_weight}")

    if use_focal_loss:
        if label_smoothing > 0:
            loss_fn = FocalLossWithLabelSmoothing(
                alpha=focal_alpha,
                gamma=focal_gamma,
                smoothing=label_smoothing,
                pos_weight=pos_weight,
            )
            if logger:
                logger.info(f"Using FocalLoss with label smoothing ({label_smoothing})")
        else:
            loss_fn = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                pos_weight=pos_weight,
            )
            if logger:
                logger.info("Using FocalLoss")
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if logger:
            logger.info("Using BCEWithLogitsLoss")

    return loss_fn


def calculate_loss(
    loss_fn: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Calculate loss based on task type.

    Args:
        loss_fn: Loss function to use
        logits: Model output logits
        targets: Ground truth targets

    Returns:
        Calculated loss tensor
    """
    targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()
    return loss_fn(logits, targets)


def get_predictions(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert logits to predictions using sigmoid.

    Args:
        logits: Model output logits

    Returns:
        Predictions (probability values)
    """
    return torch.sigmoid(logits)


def prepare_targets_for_metrics(targets: torch.Tensor) -> torch.Tensor:
    """
    Prepare targets for metric computation.

    Args:
        targets: Ground truth targets

    Returns:
        Targets prepared for metric computation
    """
    return targets.int().unsqueeze(1) if targets.dim() == 1 else targets.int()


def get_model_summary(
    model: torch.nn.Module,
    config: "ConfigManager",
    lightning_module_name: str,
    use_focal_loss: bool,
    label_smoothing: float,
    focal_alpha: float,
    focal_gamma: float,
    unfrozen_layers: int,
    device: torch.device
) -> Dict[str, Any]:
    """
    Get comprehensive model summary.

    Args:
        model: The PyTorch model
        config: ConfigManager instance
        lightning_module_name: Name of the lightning module
        use_focal_loss: Whether Focal Loss is being used
        label_smoothing: Label smoothing factor
        focal_alpha: Alpha parameter for Focal Loss
        focal_gamma: Gamma parameter for Focal Loss
        unfrozen_layers: Number of unfrozen layers
        device: Device the model is on

    Returns:
        Dictionary containing model summary information
    """
    model_info = model.get_model_info()
    model_info.update({
        "lightning_module": lightning_module_name,
        "optimizer": "AdamW",
        "learning_rate": config.get("experiment.learning_rate", 0),
        "weight_decay": config.get("experiment.weight_decay", 0),
        "scheduler": "CosineAnnealingWarmRestarts" if use_focal_loss else "ReduceLROnPlateau",
        "loss_function": "FocalLoss" if use_focal_loss else "BCEWithLogitsLoss",
        "label_smoothing": label_smoothing,
        "focal_alpha": focal_alpha if use_focal_loss else None,
        "focal_gamma": focal_gamma if use_focal_loss else None,
        "unfrozen_layers": unfrozen_layers,
        "device": str(device)
    })
    return model_info
