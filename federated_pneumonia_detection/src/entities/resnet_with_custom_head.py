"""
ResNet50 V2 with custom classification head for pneumonia detection.
Provides configurable backbone freezing and fine-tuning capabilities.
"""

from typing import Optional, TYPE_CHECKING
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

from federated_pneumonia_detection.src.internals.loggers.logger import get_logger
from .res_internals.validation import validate_parameters
from .res_internals.model_builder import (
    create_backbone,
    create_classifier_head,
    configure_fine_tuning,
)
from .res_internals.model_ops import (
    get_model_info as get_info,
    freeze_backbone as freeze_bb,
    unfreeze_backbone as unfreeze_bb,
    set_dropout_rate as set_dropout,
    get_feature_maps as extract_features,
)


class ResNetWithCustomHead(nn.Module):
    """
    ResNet50 V2 backbone with custom binary classification head.

    Features configurable fine-tuning, dropout, and head architecture
    with comprehensive error handling and validation.
    """

    def __init__(
        self,
        config: Optional["ConfigManager"] = None,
        base_model_weights: Optional[ResNet50_Weights] = None,
        num_classes: int = 1,
        dropout_rate: Optional[float] = None,
        fine_tune_layers_count: Optional[int] = None,
        custom_head_sizes: Optional[list] = None,
    ):
        """
        Initialize ResNet50 with custom classification head.

        Args:
            config: ConfigManager for configuration
            base_model_weights: Optional weights for ResNet50 backbone
            num_classes: Number of output classes (1 for binary classification)
            dropout_rate: Optional dropout rate override
            fine_tune_layers_count: Optional fine-tuning layers override
            custom_head_sizes: Optional custom head architecture [2048, 256, 64, num_classes]

        Raises:
            ValueError: If configuration parameters are invalid
        """
        super().__init__()
        if config is None:
            from federated_pneumonia_detection.config.config_manager import (
                ConfigManager,
            )

            config = ConfigManager()

        self.config = config
        self.logger = get_logger(__name__)

        # Set parameters from config with overrides
        self.num_classes = num_classes
        self.dropout_rate = (
            dropout_rate
            if dropout_rate is not None
            else config.get("experiment.dropout_rate", 0.5)
        )
        self.fine_tune_layers_count = (
            fine_tune_layers_count
            if fine_tune_layers_count is not None
            else config.get("experiment.fine_tune_layers_count", 0)
        )

        # Validate parameters
        validate_parameters(
            self.num_classes,
            self.dropout_rate,
            self.fine_tune_layers_count,
            self.logger,
        )

        # Set up backbone weights
        if base_model_weights is None:
            base_model_weights = ResNet50_Weights.IMAGENET1K_V2

        self.base_model_weights = base_model_weights

        # Initialize backbone
        self.features, self.backbone_layers = create_backbone(
            self.base_model_weights, self.logger
        )

        # Initialize classifier head
        self.classifier = create_classifier_head(
            self.num_classes, self.dropout_rate, custom_head_sizes, self.logger
        )

        # Apply fine-tuning configuration
        configure_fine_tuning(self.features, self.fine_tune_layers_count, self.logger)

        self.logger.info(
            f"ResNetWithCustomHead initialized: "
            f"classes={self.num_classes}, dropout={self.dropout_rate}, "
            f"fine_tune_layers={self.fine_tune_layers_count}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)

        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            # Extract features using ResNet50 backbone
            features = self.features(x)

            # Apply classification head
            output = self.classifier(features)

            return output

        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            raise RuntimeError(f"Forward pass failed: {e}")

    def get_model_info(self) -> dict:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model statistics and configuration
        """
        return get_info(
            self.features,
            self.classifier,
            self.base_model_weights,
            self.num_classes,
            self.dropout_rate,
            self.fine_tune_layers_count,
            self.config,
        )

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        freeze_bb(self.features, self.logger)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        unfreeze_bb(self.features, self.logger)

    def set_dropout_rate(self, new_rate: float) -> None:
        """
        Update dropout rate in classifier head.

        Args:
            new_rate: New dropout rate (0.0 to 1.0)
        """
        self.dropout_rate = set_dropout(self.classifier, new_rate, self.logger)

    def get_feature_maps(
        self, x: torch.Tensor, layer_name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract feature maps from a specific layer.

        Args:
            x: Input tensor
            layer_name: Optional layer name (if None, returns final features)

        Returns:
            Feature maps tensor
        """
        return extract_features(self.features, x, layer_name, self.logger)

    def _unfreeze_last_n_layers(self, n_layers: int) -> None:
        """
        Unfreeze the last n layers of the backbone.

        Args:
            n_layers: Number of layers to unfreeze from the end

        Note:
            This is a progressive unfreezing method used by callbacks.
        """
        from .res_internals.model_builder import unfreeze_last_n_layers

        unfreeze_last_n_layers(self.features, n_layers, self.logger)
