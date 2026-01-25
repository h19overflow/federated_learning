import pytest
from unittest.mock import MagicMock, patch
from federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced import (
    LitResNetEnhanced,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.losses.focal_loss import (
    FocalLossWithLabelSmoothing,
)


class TestModelExceptions:
    """Test suite for exception handling in models and loss functions."""

    @patch(
        "federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced.ResNetWithCustomHead"
    )
    def test_init_invalid_learning_rate(self, mock_resnet, mock_config):
        """Test initialization with invalid learning rate."""
        # Case 1: lr = 0
        mock_config.get.side_effect = (
            lambda key, default=None: 0
            if key == "experiment.learning_rate"
            else default
        )
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            LitResNetEnhanced(config=mock_config)

        # Case 2: lr = -1
        mock_config.get.side_effect = (
            lambda key, default=None: -1
            if key == "experiment.learning_rate"
            else default
        )
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            LitResNetEnhanced(config=mock_config)

    @patch(
        "federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced.ResNetWithCustomHead"
    )
    def test_init_invalid_weight_decay(self, mock_resnet, mock_config):
        """Test initialization with invalid weight decay."""

        # Case: weight_decay = -1e-4
        def get_config(key, default=None):
            if key == "experiment.learning_rate":
                return 0.001
            if key == "experiment.weight_decay":
                return -1e-4
            return default

        mock_config.get.side_effect = get_config
        with pytest.raises(ValueError, match="Weight decay must be non-negative"):
            LitResNetEnhanced(config=mock_config)

    def test_invalid_smoothing_too_high(self):
        """Test FocalLossWithLabelSmoothing with smoothing >= 0.5."""
        # Case 1: smoothing = 0.5
        with pytest.raises(ValueError, match=r"Smoothing must be in \[0, 0.5\)"):
            FocalLossWithLabelSmoothing(smoothing=0.5)

        # Case 2: smoothing = 0.6
        with pytest.raises(ValueError, match=r"Smoothing must be in \[0, 0.5\)"):
            FocalLossWithLabelSmoothing(smoothing=0.6)

    def test_invalid_smoothing_negative(self):
        """Test FocalLossWithLabelSmoothing with smoothing < 0."""
        # Case: smoothing = -0.1
        with pytest.raises(ValueError, match=r"Smoothing must be in \[0, 0.5\)"):
            FocalLossWithLabelSmoothing(smoothing=-0.1)
