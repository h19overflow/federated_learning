"""
ResNet50 V2 with custom classification head for pneumonia detection.
Provides configurable backbone freezing and fine-tuning capabilities.
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger


class ResNetWithCustomHead(nn.Module):
    """
    ResNet50 V2 backbone with custom binary classification head.

    Features configurable fine-tuning, dropout, and head architecture
    with comprehensive error handling and validation.
    """

    def __init__(
        self,
        constants: SystemConstants,
        config: ExperimentConfig,
        base_model_weights: Optional[ResNet50_Weights] = None,
        num_classes: int = 1,
        dropout_rate: Optional[float] = None,
        fine_tune_layers_count: Optional[int] = None,
        custom_head_sizes: Optional[list] = None,
    ):
        """
        Initialize ResNet50 with custom classification head.

        Args:
            constants: SystemConstants for configuration
            config: ExperimentConfig for model parameters
            base_model_weights: Optional weights for ResNet50 backbone
            num_classes: Number of output classes (1 for binary classification)
            dropout_rate: Optional dropout rate override
            fine_tune_layers_count: Optional fine-tuning layers override
            custom_head_sizes: Optional custom head architecture [2048, 256, 64, num_classes]

        Raises:
            ValueError: If configuration parameters are invalid
        """
        super().__init__()

        self.constants = constants
        self.config = config
        self.logger = get_logger(__name__)

        # Set parameters from config with overrides
        self.num_classes = num_classes
        self.dropout_rate = (
            dropout_rate
            if dropout_rate is not None
            else getattr(config, "dropout_rate", 0.5)
        )
        self.fine_tune_layers_count = (
            fine_tune_layers_count
            if fine_tune_layers_count is not None
            else getattr(config, "fine_tune_layers_count", 0)
        )

        # Validate parameters
        self._validate_parameters()

        # Set up backbone weights
        if base_model_weights is None:
            base_model_weights = ResNet50_Weights.IMAGENET1K_V2

        self.base_model_weights = base_model_weights

        # Initialize backbone
        self._create_backbone()

        # Initialize classifier head
        self._create_classifier_head(custom_head_sizes)

        # Apply fine-tuning configuration
        self._configure_fine_tuning()

        self.logger.info(
            f"ResNetWithCustomHead initialized: "
            f"classes={self.num_classes}, dropout={self.dropout_rate}, "
            f"fine_tune_layers={self.fine_tune_layers_count}"
        )

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.num_classes <= 0:
            self.logger.error("num_classes must be positive")
            raise ValueError("num_classes must be positive")

        if not 0.0 <= self.dropout_rate <= 1.0:
            self.logger.error("dropout_rate must be between 0.0 and 1.0")
            raise ValueError("dropout_rate must be between 0.0 and 1.0")

        if not isinstance(self.fine_tune_layers_count, int):
            self.logger.error("fine_tune_layers_count must be an integer")
            raise ValueError("fine_tune_layers_count must be an integer")

    def _create_backbone(self) -> None:
        """Create and configure the ResNet50 backbone."""
        try:
            # Load pretrained ResNet50
            base_model = models.resnet50(weights=self.base_model_weights)

            # Extract feature layers (remove avgpool and fc)
            self.features = nn.Sequential(*list(base_model.children())[:-2])

            # Store backbone info for fine-tuning
            self.backbone_layers = list(self.features.children())

            self.logger.info(
                f"ResNet50 backbone created with {len(self.backbone_layers)} layers"
            )

        except Exception as e:
            self.logger.error(f"Failed to create ResNet50 backbone: {e}")
            raise RuntimeError(f"Failed to create ResNet50 backbone: {e}")

    def _create_classifier_head(self, custom_head_sizes: Optional[list] = None) -> None:
        """Create the custom classification head."""
        # Default head architecture: 2048 -> 256 -> 64 -> num_classes
        if custom_head_sizes is None:
            head_sizes = [2048, 256, 64, self.num_classes]
        else:
            if len(custom_head_sizes) < 2:
                self.logger.error("custom_head_sizes must have at least 2 elements")
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
                    [nn.ReLU(inplace=True), nn.Dropout(self.dropout_rate)]
                )

        self.classifier = nn.Sequential(*classifier_layers)

        self.logger.info(
            f"Classifier head created: {' -> '.join(map(str, head_sizes))}"
        )

    def _configure_fine_tuning(self) -> None:
        """Configure fine-tuning of backbone layers."""
        # Freeze all backbone parameters by default
        for param in self.features.parameters():
            param.requires_grad = False

        total_frozen = sum(1 for param in self.features.parameters())

        # Handle fine-tuning configuration
        if self.fine_tune_layers_count < 0:
            # Unfreeze the last N layers
            self._unfreeze_last_n_layers(abs(self.fine_tune_layers_count))

        # Count unfrozen parameters
        total_unfrozen = sum(
            1 for param in self.features.parameters() if param.requires_grad
        )

        self.logger.info(
            f"Fine-tuning: {total_unfrozen}/{total_frozen + total_unfrozen} backbone parameters unfrozen"
        )

    def _unfreeze_last_n_layers(self, n_layers: int) -> None:
        """Unfreeze the last n layers of the backbone."""
        # Get parameter-containing layers in reverse order
        param_layers = []
        for module in reversed(list(self.features.modules())):
            if any(param.numel() > 0 for param in module.parameters(recurse=False)):
                param_layers.append(module)

        # Unfreeze the last n layers
        layers_to_unfreeze = min(n_layers, len(param_layers))
        for i, layer in enumerate(param_layers):
            if i < layers_to_unfreeze:
                for param in layer.parameters(recurse=False):
                    param.requires_grad = True

        self.logger.info(
            f"Unfroze last {layers_to_unfreeze} parameter-containing layers"
        )

        # Unfreeze the first n layers
        layers_to_unfreeze = min(n_layers, len(param_layers))
        for i, layer in enumerate(param_layers):
            if i < layers_to_unfreeze:
                for param in layer.parameters(recurse=False):
                    param.requires_grad = True

        self.logger.info(
            f"Unfroze first {layers_to_unfreeze} parameter-containing layers"
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
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.features.parameters())
        backbone_trainable = sum(
            p.numel() for p in self.features.parameters() if p.requires_grad
        )
        head_params = sum(p.numel() for p in self.classifier.parameters())

        return {
            "model_name": "ResNetWithCustomHead",
            "backbone": "ResNet50",
            "backbone_weights": str(self.base_model_weights),
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "fine_tune_layers_count": self.fine_tune_layers_count,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_parameters": backbone_params,
            "backbone_trainable_parameters": backbone_trainable,
            "head_parameters": head_params,
            "trainable_ratio": trainable_params / total_params
            if total_params > 0
            else 0.0,
            "input_size": self.constants.IMG_SIZE,
            "architecture": str(self.classifier),
        }

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = False
        self.logger.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = True
        self.logger.info("Backbone unfrozen")

    def set_dropout_rate(self, new_rate: float) -> None:
        """
        Update dropout rate in classifier head.

        Args:
            new_rate: New dropout rate (0.0 to 1.0)
        """
        if not 0.0 <= new_rate <= 1.0:
            self.logger.error("Dropout rate must be between 0.0 and 1.0")
            raise ValueError("Dropout rate must be between 0.0 and 1.0")

        for module in self.classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_rate

        self.dropout_rate = new_rate
        self.logger.info(f"Dropout rate updated to {new_rate}")

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
        if layer_name is None:
            return self.features(x)

        # Hook-based feature extraction for specific layers
        features = {}

        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output

            return hook

        # Register hook
        for name, module in self.features.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn(name))
                break
        else:
            self.logger.error(f"Layer '{layer_name}' not found")
            raise ValueError(f"Layer '{layer_name}' not found")

        # Forward pass
        _ = self.features(x)

        # Remove hook
        handle.remove()

        if layer_name not in features:
            self.logger.error(f"Failed to extract features from layer '{layer_name}'")
            raise RuntimeError(f"Failed to extract features from layer '{layer_name}'")

        return features[layer_name]
