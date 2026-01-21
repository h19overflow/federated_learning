"""Shared fixtures for inference endpoint tests.

Provides mock models, test clients, and image fixtures for testing
inference endpoints (single, batch, GradCAM, health checks).
"""

import base64
import io
import sys
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from PIL import Image

# Patch circular imports before loading endpoints
sys.modules["federated_pneumonia_detection.src.boundary.models"] = MagicMock()
sys.modules["federated_pneumonia_detection.src.boundary.models.Base"] = MagicMock()

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    ClinicalInterpretation,
    InferencePrediction,
    PredictionClass,
    RiskAssessment,
    SingleImageResult,
)
from federated_pneumonia_detection.src.control.model_inferance import InferenceService

# =============================================================================
# Mock Model Fixtures
# =============================================================================


@pytest.fixture
def mock_pytorch_model():
    """Create mock PyTorch model for inference testing."""
    model = MagicMock()
    model.eval = Mock(return_value=None)
    model.to = Mock(return_value=model)
    model.state_dict = Mock(return_value={})

    # Mock forward pass output
    mock_output = MagicMock()
    mock_output.detach.return_value.cpu.return_value.numpy.return_value = np.array(
        [[0.3, 0.7]],  # [normal_prob, pneumonia_prob]
    )
    model.__call__ = Mock(return_value=mock_output)

    # Add common model attributes
    model.model_version = "test_model_v1.0"

    return model


@pytest.fixture
def mock_inference_engine(mock_pytorch_model):
    """Create mock InferenceEngine with loaded model."""
    engine = MagicMock()
    engine.model = mock_pytorch_model
    engine.model_version = "test_model_v1.0"
    engine.device = "cpu"

    # Mock prediction method
    engine.predict = Mock(return_value=("PNEUMONIA", 0.95, 0.7, 0.3))

    # Mock preprocess method
    engine.preprocess = Mock(
        return_value=np.random.randn(1, 3, 224, 224).astype(np.float32),
    )

    # Mock get_info method
    engine.get_info = Mock(
        return_value={
            "model_loaded": True,
            "gpu_available": False,
            "model_version": "test_model_v1.0",
            "device": "cpu",
        },
    )

    return engine


@pytest.fixture
def mock_clinical_interpreter():
    """Create mock clinical interpreter agent."""
    interpreter = MagicMock()

    # Create sample clinical interpretation
    sample_interpretation = ClinicalInterpretation(
        summary="Findings suggest possible pneumonia in the lung fields.",
        confidence_explanation="Model confidence is high (95%), indicating clear patterns.",
        risk_assessment=RiskAssessment(
            risk_level="MODERATE",
            false_negative_risk="LOW",
            factors=["Elevated opacity in right lung", "Clear heart borders"],
        ),
        recommendations=[
            "Confirm with clinical correlation",
            "Consider follow-up imaging if symptoms persist",
        ],
    )

    interpreter.generate = AsyncMock(return_value=sample_interpretation)
    return interpreter


@pytest.fixture
def mock_inference_service(mock_inference_engine, mock_clinical_interpreter):
    """Create fully mocked InferenceService."""
    service = MagicMock(spec=InferenceService)

    # Mock core properties
    service.engine = mock_inference_engine
    service.interpreter = mock_clinical_interpreter

    # Mock methods
    service.check_ready_or_raise = Mock(return_value=None)
    service.predict = Mock(return_value=("PNEUMONIA", 0.95, 0.7, 0.3))
    service.create_prediction = Mock(
        return_value=InferencePrediction(
            predicted_class=PredictionClass.PNEUMONIA,
            confidence=0.95,
            pneumonia_probability=0.7,
            normal_probability=0.3,
        ),
    )

    # Mock process_single for batch tests
    service.process_single = AsyncMock(
        return_value=SingleImageResult(
            filename="test.jpg",
            success=True,
            prediction=InferencePrediction(
                predicted_class=PredictionClass.PNEUMONIA,
                confidence=0.95,
                pneumonia_probability=0.7,
                normal_probability=0.3,
            ),
            clinical_interpretation=None,
            processing_time_ms=50.0,
        ),
    )

    # Mock batch stats
    service.batch_stats = MagicMock()
    service.batch_stats.calculate = Mock(
        return_value={
            "total_images": 3,
            "successful": 3,
            "failed": 0,
            "normal_count": 1,
            "pneumonia_count": 2,
            "avg_confidence": 0.9,
            "avg_processing_time_ms": 45.0,
            "high_risk_count": 0,
        },
    )

    # Mock health info
    service.get_info = Mock(
        return_value={
            "status": "healthy",
            "model_loaded": True,
            "gpu_available": False,
            "model_version": "test_model_v1.0",
        },
    )

    # Mock logger
    service.logger = MagicMock()
    service.logger.log_single = Mock()
    service.logger.log_batch = Mock()
    service.logger.log_error = Mock()

    # Mock validator
    service.validator = MagicMock()
    service.validator.validate_or_raise = Mock()

    # Mock processor
    service.processor = MagicMock()

    return service


@pytest.fixture
def mock_inference_service_unavailable():
    """Create mock InferenceService with unavailable model."""
    service = MagicMock(spec=InferenceService)
    service.engine = None
    service.is_ready = Mock(return_value=False)
    service.get_info = Mock(
        return_value={
            "status": "unhealthy",
            "model_loaded": False,
            "gpu_available": False,
            "model_version": None,
        },
    )
    service.check_ready_or_raise = Mock(
        side_effect=Exception("Inference model is not available"),
    )
    return service


# =============================================================================
# Test Image Fixtures
# =============================================================================


@pytest.fixture
def sample_xray_image() -> Image.Image:
    """Create a sample X-ray-like image for testing."""
    # Create grayscale image with X-ray-like characteristics
    data = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
    return Image.fromarray(data, mode="L")


@pytest.fixture
def sample_xray_rgb() -> Image.Image:
    """Create a sample RGB X-ray-like image for testing."""
    data = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(data, mode="RGB")


@pytest.fixture
def sample_xray_png_bytes(sample_xray_image) -> bytes:
    """Create PNG bytes from sample X-ray image."""
    buffer = io.BytesIO()
    sample_xray_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def sample_xray_jpg_bytes(sample_xray_rgb) -> bytes:
    """Create JPEG bytes from sample X-ray image."""
    buffer = io.BytesIO()
    sample_xray_rgb.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def mock_upload_file(sample_xray_png_bytes, filename="test_xray.png") -> UploadFile:
    """Create a mock UploadFile object for testing."""
    file = MagicMock(spec=UploadFile)
    file.filename = filename
    file.content_type = "image/png"
    file.file = io.BytesIO(sample_xray_png_bytes)
    file.read = Mock(return_value=sample_xray_png_bytes)
    return file


@pytest.fixture
def mock_upload_file_invalid():
    """Create a mock invalid upload file for error testing."""
    file = MagicMock(spec=UploadFile)
    file.filename = "invalid.txt"
    file.content_type = "text/plain"
    file.read = Mock(return_value=b"invalid content")
    return file


@pytest.fixture
def mock_upload_file_large():
    """Create a mock oversized upload file for size limit testing."""
    # Create 11MB file (over typical 10MB limit)
    large_data = b"x" * (11 * 1024 * 1024)
    file = MagicMock(spec=UploadFile)
    file.filename = "large_xray.png"
    file.content_type = "image/png"
    file.read = Mock(return_value=large_data)
    return file


@pytest.fixture
def mock_upload_files_list(mock_upload_file, count=3) -> List[UploadFile]:
    """Create a list of mock upload files for batch testing."""
    files = []
    for i in range(count):
        file = MagicMock(spec=UploadFile)
        file.filename = f"xray_{i:03d}.png"
        file.content_type = "image/png"

        # Create unique image data for each file
        data = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
        img = Image.fromarray(data, mode="L")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        file_bytes = buffer.read()

        file.file = io.BytesIO(file_bytes)
        file.read = Mock(return_value=file_bytes)
        files.append(file)
    return files


@pytest.fixture
def mock_upload_files_large_batch():
    """Create a list exceeding batch size limit (501 files)."""
    files = []
    for i in range(501):
        file = MagicMock(spec=UploadFile)
        file.filename = f"xray_{i:03d}.png"
        file.content_type = "image/png"
        file.read = Mock(return_value=b"fake image data")
        files.append(file)
    return files


# =============================================================================
# Test Client Fixtures
# =============================================================================


@pytest.fixture
def test_inference_client(mock_inference_service):
    """Create test client for inference endpoints with mocked dependencies."""
    from fastapi import FastAPI

    from federated_pneumonia_detection.src.api.endpoints.inference.batch_prediction_endpoints import (
        router as batch_router,
    )
    from federated_pneumonia_detection.src.api.endpoints.inference.gradcam_endpoints import (
        router as gradcam_router,
    )
    from federated_pneumonia_detection.src.api.endpoints.inference.health_endpoints import (
        router as health_router,
    )

    # Import routers after patching
    from federated_pneumonia_detection.src.api.endpoints.inference.single_prediction_endpoint import (
        router as single_router,
    )

    app = FastAPI()

    # Patch dependency in deps module AND in each endpoint module that imports it
    with (
        patch(
            "federated_pneumonia_detection.src.api.deps.get_inference_service",
            return_value=mock_inference_service,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.inference.single_prediction_endpoint.get_inference_service",
            return_value=mock_inference_service,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.inference.batch_prediction_endpoints.get_inference_service",
            return_value=mock_inference_service,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.inference.health_endpoints.get_inference_service",
            return_value=mock_inference_service,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.inference.gradcam_endpoints.get_inference_service",
            return_value=mock_inference_service,
        ),
    ):
        app.include_router(single_router)
        app.include_router(batch_router)
        app.include_router(health_router)
        app.include_router(gradcam_router)

    return TestClient(app)


@pytest.fixture
def test_unavailable_client(mock_inference_service_unavailable):
    """Create test client with unavailable model for error testing."""
    from fastapi import FastAPI

    from federated_pneumonia_detection.src.api.endpoints.inference.health_endpoints import (
        router as health_router,
    )

    # Import routers after patching
    from federated_pneumonia_detection.src.api.endpoints.inference.single_prediction_endpoint import (
        router as single_router,
    )

    app = FastAPI()

    with (
        patch(
            "federated_pneumonia_detection.src.api.deps.get_inference_service",
            return_value=mock_inference_service_unavailable,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.inference.single_prediction_endpoint.get_inference_service",
            return_value=mock_inference_service_unavailable,
        ),
        patch(
            "federated_pneumonia_detection.src.api.endpoints.inference.health_endpoints.get_inference_service",
            return_value=mock_inference_service_unavailable,
        ),
    ):
        app.include_router(single_router)
        app.include_router(health_router)

    return TestClient(app)


# =============================================================================
# Helper Functions
# =============================================================================


def create_base64_image(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def assert_valid_inference_response(response_json: dict) -> None:
    """Assert that inference response has valid structure."""
    assert "success" in response_json
    assert "prediction" in response_json
    assert "processing_time_ms" in response_json

    pred = response_json["prediction"]
    assert "predicted_class" in pred
    assert "confidence" in pred
    assert "pneumonia_probability" in pred
    assert "normal_probability" in pred

    assert 0 <= pred["confidence"] <= 1
    assert 0 <= pred["pneumonia_probability"] <= 1
    assert 0 <= pred["normal_probability"] <= 1


def assert_valid_heatmap_response(response_json: dict) -> None:
    """Assert that heatmap response has valid structure."""
    assert "success" in response_json
    assert "heatmap_base64" in response_json
    assert "original_image_base64" in response_json
    assert "processing_time_ms" in response_json
    assert response_json["heatmap_base64"].startswith("data:image/png;base64,")
    assert response_json["original_image_base64"].startswith("data:image/png;base64,")


# =============================================================================
# Parametrized Test Data
# =============================================================================


@pytest.fixture(
    params=[
        ("jet", 0.4),
        ("hot", 0.5),
        ("viridis", 0.3),
        ("jet", 0.1),
        ("jet", 0.9),
    ],
)
def heatmap_params(request):
    """Parametrized heatmap generation parameters."""
    colormap, alpha = request.param
    return {"colormap": colormap, "alpha": alpha}


@pytest.fixture(
    params=[
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("JPG", "image/jpeg"),
    ],
)
def valid_image_formats(request):
    """Parametrized valid image formats for testing."""
    extension, content_type = request.param
    return {"extension": extension, "content_type": content_type}


@pytest.fixture(
    params=[
        "test_xray.png",
        "chest_xray.jpg",
        "patient_scan.jpeg",
        "X-RAY.PNG",  # Test case insensitivity
    ],
)
def sample_filenames(request):
    """Parametrized sample filenames for testing."""
    return request.param
