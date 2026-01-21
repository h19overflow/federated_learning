"""Model operation utilities for ResNetWithCustomHead."""

from typing import Any, Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights


def get_model_info(
    features: nn.Sequential,
    classifier: nn.Sequential,
    base_model_weights: ResNet50_Weights,
    num_classes: int,
    dropout_rate: float,
    fine_tune_layers_count: int,
    config: Any,
) -> dict:
    """
    Get comprehensive model information.

    Args:
        features: Backbone features
        classifier: Classifier head
        base_model_weights: Backbone weights
        num_classes: Number of classes
        dropout_rate: Dropout rate
        fine_tune_layers_count: Fine-tuning layers count
        config: ConfigManager instance

    Returns:
        Dictionary with model statistics and configuration
    """
    # Calculate combined parameters
    all_params = list(features.parameters()) + list(classifier.parameters())
    total_params = sum(p.numel() for p in all_params)
    trainable_params = sum(p.numel() for p in all_params if p.requires_grad)

    backbone_params = sum(p.numel() for p in features.parameters())
    backbone_trainable = sum(
        p.numel() for p in features.parameters() if p.requires_grad
    )
    head_params = sum(p.numel() for p in classifier.parameters())

    return {
        "model_name": "ResNetWithCustomHead",
        "backbone": "ResNet50",
        "backbone_weights": str(base_model_weights),
        "num_classes": num_classes,
        "dropout_rate": dropout_rate,
        "fine_tune_layers_count": fine_tune_layers_count,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "backbone_parameters": backbone_params,
        "backbone_trainable_parameters": backbone_trainable,
        "head_parameters": head_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0,
        "input_size": tuple(config.get("system.img_size", [224, 224])),
        "architecture": str(classifier),
    }


def freeze_backbone(features: nn.Sequential, logger: Any) -> None:
    """
    Freeze all backbone parameters.

    Args:
        features: Backbone features
        logger: Logger instance
    """
    for param in features.parameters():
        param.requires_grad = False
    logger.info("Backbone frozen")


def unfreeze_backbone(features: nn.Sequential, logger: Any) -> None:
    """
    Unfreeze all backbone parameters.

    Args:
        features: Backbone features
        logger: Logger instance
    """
    for param in features.parameters():
        param.requires_grad = True
    logger.info("Backbone unfrozen")


def set_dropout_rate(classifier: nn.Sequential, new_rate: float, logger: Any) -> float:
    """
    Update dropout rate in classifier head.

    Args:
        classifier: Classifier head
        new_rate: New dropout rate (0.0 to 1.0)
        logger: Logger instance

    Returns:
        The new dropout rate

    Raises:
        ValueError: If new_rate is invalid
    """
    if not 0.0 <= new_rate <= 1.0:
        logger.error("Dropout rate must be between 0.0 and 1.0")
        raise ValueError("Dropout rate must be between 0.0 and 1.0")

    for module in classifier.modules():
        if isinstance(module, nn.Dropout):
            module.p = new_rate

    logger.info(f"Dropout rate updated to {new_rate}")
    return new_rate


def get_feature_maps(
    features: nn.Sequential,
    x: torch.Tensor,
    layer_name: Optional[str],
    logger: Any,
) -> torch.Tensor:
    """
    Extract feature maps from a specific layer.

    Args:
        features: Backbone features
        x: Input tensor
        layer_name: Optional layer name (if None, returns final features)
        logger: Logger instance

    Returns:
        Feature maps tensor

    Raises:
        ValueError: If layer_name not found
        RuntimeError: If feature extraction fails
    """
    if layer_name is None:
        return features(x)

    # Hook-based feature extraction for specific layers
    feature_maps = {}

    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output

        return hook

    # Register hook
    for name, module in features.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn(name))
            break
    else:
        logger.error(f"Layer '{layer_name}' not found")
        raise ValueError(f"Layer '{layer_name}' not found")

    # Forward pass
    _ = features(x)

    # Remove hook
    handle.remove()

    if layer_name not in feature_maps:
        logger.error(f"Failed to extract features from layer '{layer_name}'")
        raise RuntimeError(f"Failed to extract features from layer '{layer_name}'")

    return feature_maps[layer_name]
