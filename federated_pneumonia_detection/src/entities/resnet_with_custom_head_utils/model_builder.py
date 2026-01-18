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

    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        custom_head_sizes: Optional custom head architecture
        logger: Logger instance

    Returns:
        Sequential classifier head

    Raises:
        ValueError: If custom_head_sizes is invalid
    """
    # Default head architecture: 2048 -> 256 -> 64 -> num_classes
    if custom_head_sizes is None:
        head_sizes = [2048, 256, 64, num_classes]
    else:
        if len(custom_head_sizes) < 2:
            logger.error("custom_head_sizes must have at least 2 elements")
            raise ValueError("custom_head_sizes must have at least 2 elements")
        head_sizes = custom_head_sizes

    # Build classifier layers
    classifier_layers: list[nn.Module] = [
        nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        nn.Flatten(),
    ]

    # Add fully connected layers with dropout
    for i in range(len(head_sizes) - 1):
        classifier_layers.extend(
            [
                nn.Linear(head_sizes[i], head_sizes[i + 1]),
            ]
        )

        # Add activation and dropout except for the final layer
        if i < len(head_sizes) - 2:
            classifier_layers.extend(
                [nn.ReLU(inplace=True), nn.Dropout(dropout_rate)]
            )

    classifier = nn.Sequential(*classifier_layers)

    logger.info(f"Classifier head created: {' -> '.join(map(str, head_sizes))}")

    return classifier


def configure_fine_tuning(features: nn.Sequential, fine_tune_layers_count: int, logger: Any) -> None:
    """
    Configure fine-tuning of backbone layers.

    Args:
        features: Backbone features sequential
        fine_tune_layers_count: Number of layers to fine-tune (negative unfreezes last n)
        logger: Logger instance
    """
    # Freeze all backbone parameters by default
    for param in features.parameters():
        param.requires_grad = False

    total_frozen = sum(1 for param in features.parameters())

    # Handle fine-tuning configuration
    if fine_tune_layers_count < 0:
        # Unfreeze the last N layers
        unfreeze_last_n_layers(features, abs(fine_tune_layers_count), logger)

    # Count unfrozen parameters
    total_unfrozen = sum(1 for param in features.parameters() if param.requires_grad)

    logger.info(
        f"Fine-tuning: {total_unfrozen}/{total_frozen + total_unfrozen} backbone parameters unfrozen"
    )


def unfreeze_last_n_layers(features: nn.Sequential, n_layers: int, logger: Any) -> None:
    """
    Unfreeze the last n layers of the backbone.

    Args:
        features: Backbone features sequential
        n_layers: Number of layers to unfreeze
        logger: Logger instance
    """
    # Get parameter-containing layers in reverse order
    param_layers = []
    for module in reversed(list(features.modules())):
        if any(param.numel() > 0 for param in module.parameters(recurse=False)):
            param_layers.append(module)

    # Unfreeze the last n layers
    layers_to_unfreeze = min(n_layers, len(param_layers))
    for i, layer in enumerate(param_layers):
        if i < layers_to_unfreeze:
            for param in layer.parameters(recurse=False):
                param.requires_grad = True

    logger.info(f"Unfroze last {layers_to_unfreeze} parameter-containing layers")

    # Unfreeze the first n layers
    layers_to_unfreeze = min(n_layers, len(param_layers))
    for i, layer in enumerate(param_layers):
        if i < layers_to_unfreeze:
            for param in layer.parameters(recurse=False):
                param.requires_grad = True

    logger.info(f"Unfroze first {layers_to_unfreeze} parameter-containing layers")
