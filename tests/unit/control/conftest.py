"""
Shared fixtures and utilities for inference module tests.
Provides mock models, test images, and common test configurations.
"""

from io import BytesIO
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi import UploadFile
from PIL import Image

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchSummaryStats,
    ClinicalInterpretation,
    InferencePrediction,
    PredictionClass,
    RiskAssessment,
    SingleImageResult,
)

# =============================================================================
# Image Fixtures
# =============================================================================


@pytest.fixture
def sample_xray_image():
    """Create a sample X-ray-like grayscale image."""
    # Create a grayscale image that resembles a chest X-ray
    img_array = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
    return Image.fromarray(img_array, mode="L")


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image."""
    img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def sample_rgba_image():
    """Create a sample RGBA image."""
    img_array = np.random.randint(0, 256, (512, 512, 4), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGBA")


@pytest.fixture
def tiny_test_image():
    """Create a tiny test image (8x8) for quick tests."""
    img_array = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def low_contrast_image():
    """Create a low contrast image for edge case testing."""
    img_array = np.full((512, 512), 128, dtype=np.uint8)
    return Image.fromarray(img_array, mode="L")


@pytest.fixture
def extreme_values_image():
    """Create image with extreme values."""
    img_array = np.zeros((512, 512), dtype=np.uint8)
    img_array[0:256, :] = 0  # Black half
    img_array[256:, :] = 255  # White half
    return Image.fromarray(img_array, mode="L")


# =============================================================================
# UploadFile Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_upload_file_jpeg():
    """Create a mock UploadFile with JPEG content."""
    img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")

    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    return UploadFile(
        filename="test_xray.jpg",
        file=buffer,
    )


@pytest.fixture
def mock_upload_file_png():
    """Create a mock UploadFile with PNG content."""
    img_array = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return UploadFile(
        filename="test_xray.png",
        file=buffer,
    )


@pytest.fixture
def mock_upload_file_invalid_type():
    """Create a mock UploadFile with invalid content type."""
    buffer = BytesIO(b"not an image")

    return UploadFile(
        filename="test.pdf",
        file=buffer,
    )


@pytest.fixture
def mock_upload_file_corrupted():
    """Create a mock UploadFile with corrupted image data."""
    buffer = BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR")

    return UploadFile(
        filename="corrupted.jpg",
        file=buffer,
    )


# =============================================================================
# PyTorch Model Fixtures
# =============================================================================


@pytest.fixture
def mock_pytorch_model():
    """Create a mock PyTorch model for testing."""
    model = Mock(spec=nn.Module)

    # Mock forward pass - binary classification output
    model.forward = Mock(return_value=torch.tensor([[0.7]]))

    # Mock eval and freeze methods
    model.eval = Mock()
    model.freeze = Mock()

    # Mock device movement
    model.to = Mock(return_value=model)

    return model


@pytest.fixture
def mock_lightning_model():
    """Create a mock Lightning model."""
    model = Mock()

    # Create a mock internal model
    base_model = Mock(spec=nn.Module)
    base_model.forward = Mock(return_value=torch.tensor([[0.7]]))

    model.model = base_model
    model.eval = Mock()
    model.freeze = Mock()
    model.to = Mock(return_value=model)

    return model


@pytest.fixture
def mock_resnet_model():
    """Create a mock ResNet model with layer4."""
    model = Mock()

    # Create mock layer structure
    layer4 = Mock(spec=nn.Conv2d)
    layer4.out_channels = 2048

    features = Mock()
    features.layer4 = layer4

    model.features = features
    model.eval = Mock()

    # Add modules for iteration
    model.modules = Mock(return_value=[layer4])

    return model


@pytest.fixture
def mock_checkpoint_path(tmp_path):
    """Create a mock checkpoint file path."""
    checkpoint_file = tmp_path / "mock_checkpoint.ckpt"
    checkpoint_file.touch()
    return checkpoint_file


# =============================================================================
# Tensor Fixtures
# =============================================================================


@pytest.fixture
def sample_image_tensor():
    """Create a sample preprocessed image tensor (1, 3, 224, 224)."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch_tensor():
    """Create a sample batch tensor (4, 3, 224, 224)."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_activation_tensor():
    """Create a sample activation tensor from a conv layer (1, 2048, 7, 7)."""
    return torch.randn(1, 2048, 7, 7)


@pytest.fixture
def sample_gradient_tensor():
    """Create a sample gradient tensor (1, 2048, 7, 7)."""
    return torch.randn(1, 2048, 7, 7)


# =============================================================================
# Prediction Schema Fixtures
# =============================================================================


@pytest.fixture
def sample_pneumonia_prediction():
    """Create a sample PNEUMONIA prediction."""
    return InferencePrediction(
        predicted_class=PredictionClass.PNEUMONIA,
        confidence=0.92,
        pneumonia_probability=0.92,
        normal_probability=0.08,
    )


@pytest.fixture
def sample_normal_prediction():
    """Create a sample NORMAL prediction."""
    return InferencePrediction(
        predicted_class=PredictionClass.NORMAL,
        confidence=0.88,
        pneumonia_probability=0.12,
        normal_probability=0.88,
    )


@pytest.fixture
def sample_uncertain_prediction():
    """Create a sample uncertain prediction (low confidence)."""
    return InferencePrediction(
        predicted_class=PredictionClass.PNEUMONIA,
        confidence=0.58,
        pneumonia_probability=0.58,
        normal_probability=0.42,
    )


@pytest.fixture
def sample_clinical_interpretation():
    """Create a sample clinical interpretation."""
    return ClinicalInterpretation(
        summary="Model detects signs consistent with pneumonia with 92.0% confidence.",
        confidence_explanation="High confidence prediction.",
        risk_assessment=RiskAssessment(
            risk_level="HIGH",
            false_negative_risk="LOW",
            factors=[
                "High confidence from validated model",
            ],
        ),
        recommendations=[
            "Immediate radiologist review recommended",
            "Consider clinical correlation with symptoms",
        ],
    )


@pytest.fixture
def sample_single_image_result():
    """Create a sample single image result."""
    return SingleImageResult(
        filename="test_xray.jpg",
        success=True,
        prediction=InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.92,
            pneumonia_probability=0.92,
            normal_probability=0.08,
        ),
        clinical_interpretation=ClinicalInterpretation(
            summary="Test summary",
            confidence_explanation="Test explanation",
            risk_assessment=RiskAssessment(
                risk_level="HIGH",
                false_negative_risk="LOW",
                factors=[],
            ),
            recommendations=[],
        ),
        processing_time_ms=150.5,
    )


@pytest.fixture
def sample_batch_summary():
    """Create a sample batch summary."""
    return BatchSummaryStats(
        total_images=10,
        successful=9,
        failed=1,
        normal_count=4,
        pneumonia_count=5,
        avg_confidence=0.85,
        avg_processing_time_ms=145.2,
        high_risk_count=3,
    )


# =============================================================================
# Batch Results Fixtures
# =============================================================================


@pytest.fixture
def sample_mixed_results():
    """Create a sample list of mixed results."""
    return [
        SingleImageResult(
            filename=f"img_{i}.jpg",
            success=i % 5 != 0,  # Every 5th fails
            prediction=(
                InferencePrediction(
                    predicted_class=PredictionClass.PNEUMONIA
                    if i % 2 == 0
                    else PredictionClass.NORMAL,
                    confidence=0.8 + (i % 3) * 0.05,
                    pneumonia_probability=0.8 + (i % 3) * 0.05
                    if i % 2 == 0
                    else 0.2 - (i % 3) * 0.05,
                    normal_probability=0.2 - (i % 3) * 0.05
                    if i % 2 == 0
                    else 0.8 + (i % 3) * 0.05,
                )
                if i % 5 != 0
                else None
            ),
            clinical_interpretation=(
                ClinicalInterpretation(
                    summary="Test",
                    confidence_explanation="Test",
                    risk_assessment=RiskAssessment(
                        risk_level="HIGH" if i % 3 == 0 else "MODERATE",
                        false_negative_risk="LOW",
                        factors=[],
                    ),
                    recommendations=[],
                )
                if i % 5 != 0 and i % 2 == 0
                else None
            ),
            error=None if i % 5 != 0 else "Processing failed",
            processing_time_ms=100.0 + i * 10,
        )
        for i in range(10)
    ]


# =============================================================================
# Mock Agent Fixture
# =============================================================================


@pytest.fixture
def mock_clinical_agent():
    """Create a mock clinical interpretation agent."""
    agent = Mock()

    async def mock_interpret(*args, **kwargs):
        """Mock async interpret method."""
        return Mock(
            summary="Agent interpretation",
            confidence_explanation="Agent explanation",
            risk_level="HIGH",
            false_negative_risk="LOW",
            risk_factors=["Agent factor"],
            recommendations=["Agent recommendation"],
        )

    agent.interpret = mock_interpret
    return agent


@pytest.fixture
def mock_clinical_agent_failing():
    """Create a mock clinical agent that raises exceptions."""
    agent = Mock()

    async def mock_interpret(*args, **kwargs):
        """Mock async interpret that raises error."""
        raise RuntimeError("Agent failed")

    agent.interpret = mock_interpret
    return agent


# =============================================================================
# Mock WandB Tracker Fixture
# =============================================================================


@pytest.fixture
def mock_wandb_tracker():
    """Create a mock W&B inference tracker."""
    tracker = Mock()
    tracker.is_active = True
    tracker.log_single_prediction = Mock()
    tracker.log_batch_prediction = Mock()
    tracker.log_error = Mock()
    return tracker


# =============================================================================
# Helper Functions
# =============================================================================


def create_mock_file(content_type: str, filename: str, data: bytes) -> UploadFile:
    """Helper to create mock UploadFile objects."""
    buffer = BytesIO(data)
    return UploadFile(
        filename=filename,
        file=buffer,
    )


def assert_valid_tensor(tensor: torch.Tensor, expected_shape: tuple):
    """Helper to assert valid tensor properties."""
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == expected_shape
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()
