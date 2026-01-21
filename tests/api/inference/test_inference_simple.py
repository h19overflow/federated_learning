"""Simplified inference tests using direct mocking."""

import io

# Patch circular imports before loading
import sys
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi import status
from PIL import Image

sys.modules["federated_pneumonia_detection.src.boundary.models"] = MagicMock()
sys.modules["federated_pneumonia_detection.src.boundary.models.Base"] = MagicMock()


class MockInferenceService:
    """Simplified mock service for testing."""

    def __init__(self):
        self.is_ready_result = True
        self.predict_result = ("PNEUMONIA", 0.95, 0.7, 0.3)
        self.engine = Mock()
        self.engine.model = Mock()
        self.engine.model_version = "test_model_v1.0"
        self.validator = Mock()
        self.validator.validate_or_raise = Mock()
        self.validator.validate = Mock(return_value=None)
        self.processor = Mock()
        self.processor.to_base64 = Mock(
            return_value="data:image/png;base64,iVBORw0KG...",
        )
        self.interpreter = AsyncMock()
        self.interpreter.generate = AsyncMock(return_value=None)
        self.batch_stats = Mock()
        self.batch_stats.calculate = Mock(
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
        self.logger = Mock()

    def check_ready_or_raise(self):
        if not self.is_ready_result:
            raise Exception("Inference model is not available")

    def predict(self, image):
        return self.predict_result

    def create_prediction(
        self,
        predicted_class,
        confidence,
        pneumonia_prob,
        normal_prob,
    ):
        from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
            InferencePrediction,
            PredictionClass,
        )

        return InferencePrediction(
            predicted_class=PredictionClass(predicted_class),
            confidence=confidence,
            pneumonia_probability=pneumonia_prob,
            normal_probability=normal_prob,
        )

    def get_info(self):
        if self.is_ready_result:
            return {
                "status": "healthy",
                "model_loaded": True,
                "gpu_available": False,
                "model_version": "test_model_v1.0",
            }
        else:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "gpu_available": False,
                "model_version": None,
            }

    async def process_single(self, file, include_clinical=False):
        from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
            InferencePrediction,
            PredictionClass,
            SingleImageResult,
        )

        return SingleImageResult(
            filename=file.filename or "unknown",
            success=True,
            prediction=InferencePrediction(
                predicted_class=PredictionClass.PNEUMONIA,
                confidence=0.95,
                pneumonia_probability=0.7,
                normal_probability=0.3,
            ),
            clinical_interpretation=None,
            processing_time_ms=50.0,
        )


@pytest.fixture
def sample_xray_bytes():
    """Create sample X-ray image bytes."""
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def mock_service():
    """Create mock inference service."""
    return MockInferenceService()


@pytest.fixture
def test_app(mock_service):
    """Create test FastAPI app with mocked service."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from federated_pneumonia_detection.src.api.endpoints.inference.batch_prediction_endpoints import (
        router as batch_router,
    )
    from federated_pneumonia_detection.src.api.endpoints.inference.health_endpoints import (
        router as health_router,
    )
    from federated_pneumonia_detection.src.api.endpoints.inference.single_prediction_endpoint import (
        router as single_router,
    )

    app = FastAPI()

    # Create a simple dependency override
    async def get_mock_service():
        return mock_service

    app.dependency_overrides[
        "federated_pneumonia_detection.src.api.endpoints.health_endpoints.get_inference_service"
    ] = get_mock_service
    app.dependency_overrides[
        "federated_pneumonia_detection.src.api.endpoints.single_prediction_endpoint.get_inference_service"
    ] = get_mock_service
    app.dependency_overrides[
        "federated_pneumonia_detection.src.api.endpoints.batch_prediction_endpoints.get_inference_service"
    ] = get_mock_service

    app.include_router(health_router)
    app.include_router(single_router)
    app.include_router(batch_router)

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_200_ok(self, test_app):
        """Test health returns 200."""
        response = test_app.get("/api/inference/health")
        assert response.status_code == status.HTTP_200_OK

    def test_health_response_structure(self, test_app):
        """Test health response structure."""
        response = test_app.get("/api/inference/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestSinglePrediction:
    """Tests for single prediction endpoint."""

    def test_predict_success(self, test_app, sample_xray_bytes):
        """Test successful prediction."""
        response = test_app.post(
            "/api/inference/predict",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "prediction" in data
        assert "processing_time_ms" in data

    def test_predict_with_clinical(self, test_app, sample_xray_bytes):
        """Test prediction with clinical interpretation."""
        response = test_app.post(
            "/api/inference/predict?include_clinical_interpretation=true",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_predict_validation(self, test_app, sample_xray_bytes):
        """Test prediction values are valid."""
        response = test_app.post(
            "/api/inference/predict",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        data = response.json()
        pred = data["prediction"]
        assert pred["predicted_class"] in ["NORMAL", "PNEUMONIA"]
        assert 0 <= pred["confidence"] <= 1
        assert 0 <= pred["pneumonia_probability"] <= 1
        assert 0 <= pred["normal_probability"] <= 1

    def test_predict_no_file(self, test_app):
        """Test prediction without file."""
        response = test_app.post("/api/inference/predict")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestBatchPrediction:
    """Tests for batch prediction endpoint."""

    def test_batch_success(self, test_app, sample_xray_bytes):
        """Test successful batch prediction."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_bytes), "image/png"))
            for i in range(3)
        ]
        response = test_app.post("/api/inference/predict-batch", files=files)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 3

    def test_batch_summary(self, test_app, sample_xray_bytes):
        """Test batch summary statistics."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_bytes), "image/png"))
            for i in range(3)
        ]
        response = test_app.post("/api/inference/predict-batch", files=files)
        data = response.json()
        assert "summary" in data
        assert data["summary"]["total_images"] == 3

    def test_batch_max_limit(self, test_app, sample_xray_bytes):
        """Test batch size limit (500)."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_bytes), "image/png"))
            for i in range(501)
        ]
        response = test_app.post("/api/inference/predict-batch", files=files)
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_batch_empty(self, test_app):
        """Test empty batch."""
        response = test_app.post("/api/inference/predict-batch", files=[])
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestErrorHandling:
    """Tests for error scenarios."""

    def test_invalid_file_format(self, test_app):
        """Test with invalid file format."""
        response = test_app.post(
            "/api/inference/predict",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        # Should fail validation
        assert response.status_code != status.HTTP_200_OK

    def test_empty_file(self, test_app):
        """Test with empty file."""
        response = test_app.post(
            "/api/inference/predict",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        # Should fail validation
        assert response.status_code != status.HTTP_200_OK


class TestImageFormats:
    """Tests for different image formats."""

    def test_png_format(self, test_app):
        """Test PNG format."""
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = test_app.post(
            "/api/inference/predict",
            files={"file": ("test.png", buffer, "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_jpeg_format(self, test_app):
        """Test JPEG format."""
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        response = test_app.post(
            "/api/inference/predict",
            files={"file": ("test.jpg", buffer, "image/jpeg")},
        )
        assert response.status_code == status.HTTP_200_OK
