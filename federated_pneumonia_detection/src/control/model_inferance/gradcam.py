"""GradCAM visualization for pneumonia detection model.

Generates class activation heatmaps to show which regions of the
chest X-ray contributed most to the model's prediction.
"""

import base64
import logging
from io import BytesIO
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ResNet models.

    Uses gradients flowing into the final convolutional layer to produce
    a coarse localization map highlighting important regions for prediction.
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[str] = None):
        """Initialize GradCAM with model and target layer.

        Args:
            model: PyTorch model (LitResNetEnhanced or similar)
            target_layer: Name of target conv layer (defaults to last conv in ResNet)
        """
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []

        # Get the actual model from Lightning wrapper if needed
        if hasattr(model, "model"):
            self.base_model = model.model
        else:
            self.base_model = model

        # Find target layer - for ResNet50, use layer4 (last conv block)
        if target_layer:
            self.target_layer = self._get_layer_by_name(target_layer)
        else:
            self.target_layer = self._get_last_conv_layer()

        self._register_hooks()

    def _get_layer_by_name(self, name: str) -> torch.nn.Module:
        """Get layer by name from model."""
        for layer_name, module in self.base_model.named_modules():
            if layer_name == name:
                return module
        raise ValueError(f"Layer {name} not found in model")

    def _get_last_conv_layer(self) -> torch.nn.Module:
        """Find the last convolutional layer in the model."""
        # For ResNetWithCustomHead, the features backbone is the ResNet
        if hasattr(self.base_model, "features"):
            # Access layer4 of the ResNet backbone
            features = self.base_model.features
            if hasattr(features, "layer4"):
                return features.layer4
            # Fallback: find last Conv2d
            last_conv = None
            for module in features.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                return last_conv

        # Generic fallback
        last_conv = None
        for module in self.base_model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module

        if last_conv is None:
            raise ValueError("No convolutional layer found in model")
        return last_conv

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate GradCAM heatmap for input.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W)
            target_class: Target class index (None = use predicted class)

        Returns:
            Heatmap as numpy array (H, W) with values 0-1
        """
        self.model.eval()

        # Enable gradients for this forward pass
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        # For binary classification with single output
        if output.shape[-1] == 1 or len(output.shape) == 1:
            # Binary: use sigmoid probability
            prob = torch.sigmoid(output)
            if target_class is None:
                # Use the predicted class direction
                target_class = 1 if prob > 0.5 else 0

            # Create scalar for backprop
            if target_class == 1:
                score = prob.squeeze()
            else:
                score = 1 - prob.squeeze()
        else:
            # Multi-class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            score = output[0, target_class]

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # Get gradients and activations
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0-1
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


def generate_heatmap_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay heatmap on original image.

    Args:
        original_image: Original PIL Image
        heatmap: Heatmap array (H, W) with values 0-1
        colormap: Matplotlib colormap name
        alpha: Overlay transparency (0-1)

    Returns:
        PIL Image with heatmap overlay
    """
    import matplotlib.pyplot as plt

    # Resize heatmap to match original image
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_resized = heatmap_resized.resize(original_image.size, Image.BILINEAR)
    heatmap_array = np.array(heatmap_resized) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_array)  # (H, W, 4) RGBA
    colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)  # RGB

    # Convert to PIL
    heatmap_img = Image.fromarray(colored_heatmap)

    # Ensure original is RGB
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    # Blend images
    overlay = Image.blend(original_image, heatmap_img, alpha)

    return overlay


def heatmap_to_base64(
    heatmap: np.ndarray,
    original_image: Image.Image,
    colormap: str = "jet",
    alpha: float = 0.4,
) -> str:
    """Convert heatmap overlay to base64 string.

    Args:
        heatmap: Raw heatmap array
        original_image: Original X-ray image
        colormap: Colormap to use
        alpha: Overlay transparency

    Returns:
        Base64-encoded PNG image string
    """
    overlay = generate_heatmap_overlay(original_image, heatmap, colormap, alpha)

    buffer = BytesIO()
    overlay.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")
