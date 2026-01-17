"""Model building utilities for ResNetWithCustomHead."""

from typing import Optional, Any
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


def create_backbone(base_model_weights: ResNet50_Weights, logger: Any) -> tuple[nn.Sequential, list]:
    """
    Create and configure the ResNet50 backbone.

    Args:
        base_model_weights: ResNet50 pretrained weights
        logger: Logger instance

    Returns:
        Tuple of (features sequential, backbone layers list)

    Raises:
        RuntimeError: If backbone creation fails
    """
    try:
        # Load pretrained ResNet50
        base_model = models.resnet50(weights=base_model_weights)

        # Extract feature layers (remove avgpool and fc)
        features = nn.Sequential(*list(base_model.children())[:-2])

        # Store backbone info for fine-tuning
        backbone_layers = list(features.children())

        logger.info(f"ResNet50 backbone created with {len(backbone_layers)} layers")

        return features, backbone_layers

    except Exception as e:
        logger.error(f"Failed to create ResNet50 backbone: {e}")
        raise RuntimeError(f"Failed to create ResNet50 backbone: {e}")


def create_classifier_head(
    num_classes: int, dropout_rate: float, custom_head_sizes: Optional[list], logger: Any
) -> nn.Sequential:
    """
    Create the custom classification head.

    Default architecture: 2048 -> 256 -> 64 -> num_classes
    """
    if custom_head_sizes is not None and len(custom_head_sizes) < 2:
        raise ValueError("custom_head_sizes must have at least 2 elements")

    head_sizes = custom_head_sizes or [2048, 256, 64, num_classes]

    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        # Layer 1: 2048 -> 256
        nn.Linear(head_sizes[0], head_sizes[1]),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        # Layer 2: 256 -> 64
        nn.Linear(head_sizes[1], head_sizes[2]),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        # Output: 64 -> num_classes
        nn.Linear(head_sizes[2], head_sizes[3]),
    )

    logger.info(f"Classifier head created: {' -> '.join(map(str, head_sizes))}")
    return classifier


def configure_fine_tuning(features: nn.Sequential, fine_tune_layers_count: int, logger: Any) -> None:
    """
    Configure fine-tuning of backbone layers.

    Args:
        fine_tune_layers_count: Negative value unfreezes last n layers (e.g., -3 unfreezes last 3)
    """
    # Freeze all backbone parameters
    for param in features.parameters():
        param.requires_grad = False

    # Unfreeze last N layers if specified
    if fine_tune_layers_count < 0:
        unfreeze_last_n_layers(features, abs(fine_tune_layers_count), logger)

    total_params = sum(1 for _ in features.parameters())
    unfrozen = sum(1 for p in features.parameters() if p.requires_grad)
    logger.info(f"Fine-tuning: {unfrozen}/{total_params} backbone parameters unfrozen")


def unfreeze_last_n_layers(features: nn.Sequential, n_layers: int, logger: Any) -> None:
    """Unfreeze the last n layers of the backbone for fine-tuning."""
    # Collect layers that have trainable parameters (in reverse order = last first)
    param_layers = [
        module
        for module in reversed(list(features.modules()))
        if any(p.numel() > 0 for p in module.parameters(recurse=False))
    ]

    layers_to_unfreeze = min(n_layers, len(param_layers))

    for layer in param_layers[:layers_to_unfreeze]:
        for param in layer.parameters(recurse=False):
            param.requires_grad = True

    logger.info(f"Unfroze last {layers_to_unfreeze} parameter-containing layers")
