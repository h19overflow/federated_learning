"""
Integration test for the handover between Training and Inference.
Verifies that a checkpoint saved by LitResNetEnhanced can be successfully
loaded and used by InferenceEngine.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# Fix for PyTorch 2.6+ weights_only=True default
from lightning_fabric.utilities.data import AttributeDict
from PIL import Image

torch.serialization.add_safe_globals([AttributeDict])

# Import the classes to test  # noqa: E402
from federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced import (  # noqa: E501, E402
    LitResNetEnhanced,
)
from federated_pneumonia_detection.src.control.model_inferance.internals.inference_engine import (  # noqa: E501, E402
    InferenceEngine,
)


class MockResNetWithCustomHead(nn.Module):
    """
    Mocked version of ResNetWithCustomHead to keep tests light.
    Matches the expected attribute structure (features, classifier).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Structure must match what's expected in state_dict if using real keys,
        # but since we create the state_dict from mock, it just needs consistency.
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def get_model_info(self):
        return {"total_parameters": 1000}

    def freeze_backbone(self):
        pass

    def unfreeze_backbone(self):
        pass

    def _unfreeze_last_n_layers(self, n):
        pass


@pytest.fixture
def mock_config():
    """Fixture for a mocked ConfigManager."""
    config = MagicMock()
    # Provide necessary config values for LitResNetEnhanced
    config_values = {
        "experiment.learning_rate": 0.001,
        "experiment.weight_decay": 0.0001,
        "experiment.dropout_rate": 0.5,
        "experiment.fine_tune_layers_count": 0,
        "experiment.epochs": 10,
        "experiment.use_torch_compile": False,
    }

    def _get_config_value(key, default=None):
        return config_values.get(key, default)

    config.get.side_effect = _get_config_value
    return config


def test_model_loading_integration(tmp_path, mock_config):
    """
    Integration test for the handover between Training (LitResNetEnhanced)
    and Inference (InferenceEngine).

    Steps:
    1. Initialize LitResNetEnhanced with a mocked backbone.
    2. Save a checkpoint to a temporary file.
    3. Initialize InferenceEngine pointing to this checkpoint.
    4. Run prediction on a dummy image.
    5. Assert that the model is loaded correctly and produces valid output.
    """
    checkpoint_path = tmp_path / "test_model_handover.ckpt"

    # 1. Setup: Initialize LitResNetEnhanced and save a checkpoint
    # We patch ResNetWithCustomHead to use our lightweight mock
    # We patch ConfigManager in its original module to affect the inline import
    with (
        patch(  # noqa: E501
            "federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced.ResNetWithCustomHead",
            MockResNetWithCustomHead,
        ),
        patch(
            "federated_pneumonia_detection.config.config_manager.ConfigManager",
            return_value=mock_config,
        ),
    ):
        model = LitResNetEnhanced(config=mock_config)

        # Create a checkpoint structure compatible with Lightning's load_from_checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "hyper_parameters": model.hparams,
            "pytorch-lightning_version": "2.0.0",
        }
        torch.save(checkpoint, checkpoint_path)

    # 2. Execution: Load the checkpoint into InferenceEngine
    # We must patch again because InferenceEngine will re-instantiate LitResNetEnhanced
    with (
        patch(  # noqa: E501
            "federated_pneumonia_detection.src.control.dl_model.internals.model.lit_resnet_enhanced.ResNetWithCustomHead",
            MockResNetWithCustomHead,
        ),
        patch(
            "federated_pneumonia_detection.config.config_manager.ConfigManager",
            return_value=mock_config,
        ),
    ):
        # Initialize InferenceEngine
        engine = InferenceEngine(checkpoint_path=checkpoint_path, device="cpu")

        # Create a dummy PIL image for prediction
        dummy_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

        # Run prediction
        result = engine.predict(dummy_img)

    # 3. Assertions
    assert engine.model is not None, "Engine should have a loaded model"
    assert not engine.model.training, "Model should be in eval mode"

    # Result structure: (predicted_class, confidence, pneumonia_prob, normal_prob)
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert result[0] in ["PNEUMONIA", "NORMAL"]
    assert 0.0 <= result[1] <= 1.0
    assert 0.0 <= result[2] <= 1.0
    assert 0.0 <= result[3] <= 1.0

    # Verify weights were correctly transferred
    # We compare the classifier weights from the original model and the loaded engine
    original_weight = model.model.classifier.weight.data
    loaded_weight = engine.model.model.classifier.weight.data
    assert torch.allclose(original_weight, loaded_weight), (
        "Loaded weights should match saved weights"
    )


if __name__ == "__main__":
    # Allow running the test directly for quick verification

    pytest.main([__file__])
