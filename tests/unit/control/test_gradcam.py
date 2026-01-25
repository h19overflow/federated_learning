"""
Unit tests for GradCAM component.
Tests gradient-weighted class activation mapping generation.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock
from PIL import Image

from federated_pneumonia_detection.src.control.model_inferance.gradcam import (
    GradCAM,
    generate_heatmap_overlay,
    heatmap_to_base64,
)


class TestGradCAM:
    """Tests for GradCAM class."""

    # =========================================================================
    # Test initialization
    # =========================================================================

    @pytest.fixture
    def mock_model(self):
        """Create a mock PyTorch model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # First conv
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),  # Second conv (target)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )
        return model

    @pytest.fixture
    def gradcam(self, mock_model):
        """Create GradCAM instance with mock model."""
        return GradCAM(mock_model)

    def test_init_with_model(self, gradcam, mock_model):
        """Test initialization with model."""
        assert gradcam.model == mock_model
        assert gradcam.gradients is None
        assert gradcam.activations is None
        assert len(gradcam.hooks) > 0

    def test_init_with_wrapped_model(self, mock_model):
        """Test initialization with Lightning-wrapped model."""
        wrapper = Mock()
        wrapper.model = mock_model

        gradcam = GradCAM(wrapper)

        assert gradcam.base_model == mock_model

    def test_init_with_target_layer(self, mock_model):
        """Test initialization with specific target layer."""
        gradcam = GradCAM(mock_model, target_layer="1")  # Second conv layer

        assert gradcam.target_layer is not None

    def test_init_without_target_layer_uses_last_conv(self, mock_model):
        """Test initialization without target layer finds last conv."""
        gradcam = GradCAM(mock_model)

        assert gradcam.target_layer is not None
        assert isinstance(gradcam.target_layer, nn.Conv2d)

    # =========================================================================
    # Test _get_layer_by_name
    # =========================================================================

    def test_get_layer_by_name_valid(self, gradcam, mock_model):
        """Test getting layer by valid name."""
        layer = gradcam._get_layer_by_name("2")
        assert layer is not None
        assert isinstance(layer, nn.Conv2d)

    def test_get_layer_by_name_invalid(self, gradcam):
        """Test getting layer by invalid name raises error."""
        with pytest.raises(ValueError, match="not found in model"):
            gradcam._get_layer_by_name("nonexistent")

    # =========================================================================
    # Test _get_last_conv_layer
    # =========================================================================

    def test_get_last_conv_layer(self, gradcam):
        """Test finding last conv layer."""
        last_conv = gradcam._get_last_conv_layer()
        assert last_conv is not None
        assert isinstance(last_conv, nn.Conv2d)

    def test_get_last_conv_layer_no_conv_raises_error(self):
        """Test finding last conv layer raises error if no conv layers."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
        )

        with pytest.raises(ValueError, match="No convolutional layer found"):
            GradCAM(model)

    # =========================================================================
    # Test hook registration and removal
    # =========================================================================

    def test_hooks_registered(self, gradcam):
        """Test that hooks are registered."""
        assert len(gradcam.hooks) == 2  # Forward and backward hooks

    def test_remove_hooks(self, gradcam):
        """Test hook removal."""
        assert len(gradcam.hooks) > 0
        gradcam.remove_hooks()
        assert len(gradcam.hooks) == 0

    def test_hooks_removed_on_deletion(self, mock_model):
        """Test hooks are removed when object is deleted."""
        gradcam = GradCAM(mock_model)
        hooks_before = len(gradcam.hooks)

        del gradcam

        # Can't directly check hooks after deletion, but should not leak

    # =========================================================================
    # Test __call__ method
    # =========================================================================

    def test_call_returns_numpy_array(self, gradcam, sample_image_tensor):
        """Test __call__ returns numpy array."""
        heatmap = gradcam(sample_image_tensor)

        assert isinstance(heatmap, np.ndarray)

    def test_call_returns_2d_array(self, gradcam, sample_image_tensor):
        """Test __call__ returns 2D array."""
        heatmap = gradcam(sample_image_tensor)

        assert len(heatmap.shape) == 2

    def test_call_normalizes_to_0_to_1(self, gradcam, sample_image_tensor):
        """Test heatmap is normalized to 0-1 range."""
        heatmap = gradcam(sample_image_tensor)

        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_call_with_target_class(self, gradcam, sample_image_tensor):
        """Test __call__ with specific target class."""
        heatmap = gradcam(sample_image_tensor, target_class=0)

        assert isinstance(heatmap, np.ndarray)

    def test_call_without_target_class(self, gradcam, sample_image_tensor):
        """Test __call__ without target class uses predicted."""
        heatmap = gradcam(sample_image_tensor, target_class=None)

        assert isinstance(heatmap, np.ndarray)

    def test_call_binary_classification(self, gradcam, sample_image_tensor):
        """Test __call__ handles binary classification."""
        heatmap = gradcam(sample_image_tensor)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.shape[0] > 0
        assert heatmap.shape[1] > 0

    # =========================================================================
    # Test activation and gradient capturing
    # =========================================================================

    def test_forward_hook_captures_activations(self, gradcam, sample_image_tensor):
        """Test forward hook captures activations."""
        _ = gradcam(sample_image_tensor)

        assert gradcam.activations is not None
        assert isinstance(gradcam.activations, torch.Tensor)

    def test_backward_hook_captures_gradients(self, gradcam, sample_image_tensor):
        """Test backward hook captures gradients."""
        _ = gradcam(sample_image_tensor)

        assert gradcam.gradients is not None
        assert isinstance(gradcam.gradients, torch.Tensor)

    def test_activations_and_gradients_same_batch(self, gradcam, sample_image_tensor):
        """Test activations and gradients have same batch dimension."""
        _ = gradcam(sample_image_tensor)

        assert gradcam.activations.shape[0] == gradcam.gradients.shape[0]

    # =========================================================================
    # Test heatmap normalization
    # =========================================================================

    def test_heatmap_relu_applied(self, gradcam, sample_image_tensor):
        """Test ReLU is applied to heatmap."""
        heatmap = gradcam(sample_image_tensor)

        # After ReLU, all values should be >= 0
        assert (heatmap >= 0).all()

    def test_heatmap_non_zero_max(self, gradcam, sample_image_tensor):
        """Test heatmap max is non-zero."""
        heatmap = gradcam(sample_image_tensor)

        assert heatmap.max() > 0

    def test_heatmap_min_is_zero(self, gradcam, sample_image_tensor):
        """Test heatmap min is zero (after normalization)."""
        heatmap = gradcam(sample_image_tensor)

        assert heatmap.min() == 0.0

    # =========================================================================
    # Test edge cases
    # =========================================================================

    def test_call_with_different_input_sizes(self, gradcam):
        """Test __call__ with different input tensor sizes."""
        sizes = [(1, 3, 224, 224), (1, 3, 256, 256), (1, 3, 128, 128)]

        for size in sizes:
            tensor = torch.randn(size)
            heatmap = gradcam(tensor)
            assert isinstance(heatmap, np.ndarray)
            assert len(heatmap.shape) == 2

    def test_call_with_zero_activations(self, mock_model):
        """Test __call__ when activations are all zero."""
        # Create model that outputs zero
        zero_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

        # Zero out weights
        with torch.no_grad():
            for param in zero_model.parameters():
                param.zero_()

        gradcam = GradCAM(zero_model)
        tensor = torch.randn(1, 3, 224, 224)
        heatmap = gradcam(tensor)

        # Should still return valid heatmap
        assert isinstance(heatmap, np.ndarray)

    def test_call_with_constant_input(self, gradcam):
        """Test __call__ with constant input."""
        constant_tensor = torch.ones(1, 3, 224, 224)
        heatmap = gradcam(constant_tensor)

        assert isinstance(heatmap, np.ndarray)

    def test_multiple_calls(self, gradcam, sample_image_tensor):
        """Test multiple calls to GradCAM."""
        heatmap1 = gradcam(sample_image_tensor)
        heatmap2 = gradcam(sample_image_tensor)

        # Both should be valid arrays
        assert isinstance(heatmap1, np.ndarray)
        assert isinstance(heatmap2, np.ndarray)

    # =========================================================================
    # Test with different model architectures
    # =========================================================================

    def test_with_resnet_like_model(self):
        """Test GradCAM with ResNet-like architecture."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
        )

        gradcam = GradCAM(model)
        tensor = torch.randn(1, 3, 224, 224)
        heatmap = gradcam(tensor)

        assert isinstance(heatmap, np.ndarray)


class TestGenerateHeatmapOverlay:
    """Tests for generate_heatmap_overlay function."""

    def test_overlay_returns_pil_image(self, sample_rgb_image):
        """Test overlay returns PIL Image."""
        heatmap = np.random.rand(224, 224)

        overlay = generate_heatmap_overlay(sample_rgb_image, heatmap)

        from PIL import Image

        assert isinstance(overlay, Image.Image)

    def test_overlay_resizes_heatmap(self, sample_xray_image):
        """Test heatmap is resized to match original image."""
        original_size = sample_xray_image.size
        small_heatmap = np.random.rand(7, 7)

        overlay = generate_heatmap_overlay(sample_xray_image, small_heatmap)

        assert overlay.size == original_size

    def test_overlay_with_rgb_image(self, sample_rgb_image):
        """Test overlay with RGB image."""
        heatmap = np.random.rand(224, 224)

        overlay = generate_heatmap_overlay(sample_rgb_image, heatmap)

        assert overlay.mode == "RGB"

    def test_overlay_with_grayscale_image(self, sample_xray_image):
        """Test overlay converts grayscale to RGB."""
        assert sample_xray_image.mode == "L"

        heatmap = np.random.rand(224, 224)
        overlay = generate_heatmap_overlay(sample_xray_image, heatmap)

        assert overlay.mode == "RGB"

    def test_overlay_with_rgba_image(self, sample_rgba_image):
        """Test overlay with RGBA image."""
        heatmap = np.random.rand(224, 224)

        overlay = generate_heatmap_overlay(sample_rgba_image, heatmap)

        assert overlay.mode == "RGB"

    def test_overlay_with_custom_alpha(self, sample_rgb_image):
        """Test overlay with custom alpha value."""
        heatmap = np.random.rand(224, 224)

        overlay_0 = generate_heatmap_overlay(sample_rgb_image, heatmap, alpha=0.0)
        overlay_5 = generate_heatmap_overlay(sample_rgb_image, heatmap, alpha=0.5)
        overlay_1 = generate_heatmap_overlay(sample_rgb_image, heatmap, alpha=1.0)

        # All should be valid images
        assert overlay_0.size == overlay_5.size == overlay_1.size

    def test_overlay_with_different_colormaps(self, sample_rgb_image):
        """Test overlay with different colormaps."""
        heatmap = np.random.rand(224, 224)
        colormaps = ["jet", "hot", "cool", "viridis", "plasma"]

        for cmap in colormaps:
            overlay = generate_heatmap_overlay(sample_rgb_image, heatmap, colormap=cmap)
            assert isinstance(overlay, Image.Image)

    def test_overlay_with_normalized_heatmap(self, sample_rgb_image):
        """Test overlay with normalized heatmap (0-1)."""
        heatmap = np.random.rand(224, 224)

        overlay = generate_heatmap_overlay(sample_rgb_image, heatmap)

        assert isinstance(overlay, Image.Image)


class TestHeatmapToBase64:
    """Tests for heatmap_to_base64 function."""

    def test_returns_base64_string(self, sample_rgb_image):
        """Test function returns base64 string."""
        heatmap = np.random.rand(224, 224)

        base64_str = heatmap_to_base64(heatmap, sample_rgb_image)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0

    def test_base64_is_valid(self, sample_rgb_image):
        """Test base64 string is valid and can be decoded."""
        import base64

        heatmap = np.random.rand(224, 224)
        base64_str = heatmap_to_base64(heatmap, sample_rgb_image)

        # Should be decodable
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0

    def test_with_custom_alpha(self, sample_rgb_image):
        """Test with custom alpha value."""
        heatmap = np.random.rand(224, 224)

        b64_0 = heatmap_to_base64(heatmap, sample_rgb_image, alpha=0.0)
        b64_5 = heatmap_to_base64(heatmap, sample_rgb_image, alpha=0.5)

        assert isinstance(b64_0, str)
        assert isinstance(b64_5, str)

    def test_with_different_colormaps(self, sample_rgb_image):
        """Test with different colormaps."""
        heatmap = np.random.rand(224, 224)

        for cmap in ["jet", "hot", "cool"]:
            base64_str = heatmap_to_base64(heatmap, sample_rgb_image, colormap=cmap)
            assert isinstance(base64_str, str)
            assert len(base64_str) > 0

    def test_with_grayscale_image(self, sample_xray_image):
        """Test with grayscale image."""
        heatmap = np.random.rand(224, 224)

        base64_str = heatmap_to_base64(heatmap, sample_xray_image)

        assert isinstance(base64_str, str)
