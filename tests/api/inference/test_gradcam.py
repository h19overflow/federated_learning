"""Tests for GradCAM endpoints."""

import io

# Patch circular imports before loading
import sys
from unittest.mock import Mock

import pytest
from fastapi import status
from PIL import Image

sys.modules["federated_pneumonia_detection.src.boundary.models"] = MagicMock()
sys.modules["federated_pneumonia_detection.src.boundary.models.Base"] = MagicMock()


@pytest.fixture
def sample_xray_bytes():
    """Create sample X-ray image bytes."""
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def mock_gradcam_service():
    """Create mock service with GradCAM support."""
    from tests.api.inference.test_inference_simple import MockInferenceService

    service = MockInferenceService()

    # Mock GradCAM components
    service.processor.to_base64 = Mock(
        return_value="data:image/png;base64,iVBORw0KG...",
    )

    # Mock the _generate_single_heatmap function's return value
    class MockHeatmapResult:
        def __init__(self):
            self.filename = "test.png"
            self.success = True
            self.heatmap_base64 = "data:image/png;base64,iVBORw0KG..."
            self.original_image_base64 = "data:image/png;base64,iVBORw0KG..."
            self.processing_time_ms = 50.0
            self.error = None

    service._heatmap_result = MockHeatmapResult()

    return service


@pytest.fixture
def test_app_gradcam(mock_gradcam_service):
    """Create test FastAPI app with mocked GradCAM service."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from federated_pneumonia_detection.src.api.endpoints.inference.gradcam_endpoints import (
        router as gradcam_router,
    )

    app = FastAPI()

    # Patch _generate_single_heatmap to return mock data
    from federated_pneumonia_detection.src.api.endpoints.inference import (
        gradcam_endpoints,
    )

    def mock_generate_heatmap(image, filename, service, colormap="jet", alpha=0.4):
        return mock_gradcam_service._heatmap_result

    gradcam_endpoints._generate_single_heatmap = mock_generate_heatmap

    async def get_mock_service():
        return mock_gradcam_service

    app.dependency_overrides[
        "federated_pneumonia_detection.src.api.endpoints.gradcam_endpoints.get_inference_service"
    ] = get_mock_service

    app.include_router(gradcam_router)

    return TestClient(app)


class TestHeatmapEndpoint:
    """Tests for single heatmap endpoint."""

    def test_heatmap_success(self, test_app_gradcam, sample_xray_bytes):
        """Test successful heatmap generation."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "heatmap_base64" in data
        assert "original_image_base64" in data

    def test_heatmap_default_params(self, test_app_gradcam, sample_xray_bytes):
        """Test heatmap with default parameters."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_colormap_jet(self, test_app_gradcam, sample_xray_bytes):
        """Test heatmap with jet colormap."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap?colormap=jet",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_colormap_hot(self, test_app_gradcam, sample_xray_bytes):
        """Test heatmap with hot colormap."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap?colormap=hot",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_colormap_viridis(self, test_app_gradcam, sample_xray_bytes):
        """Test heatmap with viridis colormap."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap?colormap=viridis",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_alpha_0_4(self, test_app_gradcam, sample_xray_bytes):
        """Test heatmap with default alpha (0.4)."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap?alpha=0.4",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_alpha_0_1(self, test_app_gradcam, sample_xray_bytes):
        """Test heatmap with minimum alpha (0.1)."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap?alpha=0.1",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_alpha_0_9(self, test_app_gradcam, sample_xray_bytes):
        """Test heatmap with maximum alpha (0.9)."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap?alpha=0.9",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_custom_params(self, test_app_gradcam, sample_xray_bytes):
        """Test heatmap with custom colormap and alpha."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap?colormap=hot&alpha=0.6",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_base64_format(self, test_app_gradcam, sample_xray_bytes):
        """Test that heatmap is base64 encoded."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        data = response.json()
        assert data["heatmap_base64"].startswith("data:image/png;base64,")
        assert data["original_image_base64"].startswith("data:image/png;base64,")

    def test_heatmap_no_file(self, test_app_gradcam):
        """Test heatmap without file."""
        response = test_app_gradcam.post("/api/inference/heatmap")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_heatmap_invalid_alpha_low(self, test_app_gradcam, sample_xray_bytes):
        """Test heatmap with alpha below minimum."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap?alpha=0.05",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_heatmap_invalid_alpha_high(self, test_app_gradcam, sample_xray_bytes):
        """Test heatmap with alpha above maximum."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap?alpha=1.0",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_heatmap_invalid_alpha_negative(self, test_app_gradcam, sample_xray_bytes):
        """Test heatmap with negative alpha."""
        response = test_app_gradcam.post(
            "/api/inference/heatmap?alpha=-0.5",
            files={"file": ("test.png", io.BytesIO(sample_xray_bytes), "image/png")},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestBatchHeatmapEndpoint:
    """Tests for batch heatmap endpoint."""

    def test_batch_heatmap_success(self, test_app_gradcam, sample_xray_bytes):
        """Test successful batch heatmap generation."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_bytes), "image/png"))
            for i in range(3)
        ]
        response = test_app_gradcam.post("/api/inference/heatmap-batch", files=files)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 3

    def test_batch_heatmap_all_success(self, test_app_gradcam, sample_xray_bytes):
        """Test batch heatmap where all succeed."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_bytes), "image/png"))
            for i in range(3)
        ]
        response = test_app_gradcam.post("/api/inference/heatmap-batch", files=files)
        data = response.json()
        assert all(result["success"] for result in data["results"])

    def test_batch_heatmap_with_params(self, test_app_gradcam, sample_xray_bytes):
        """Test batch heatmap with custom parameters."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_bytes), "image/png"))
            for i in range(2)
        ]
        response = test_app_gradcam.post(
            "/api/inference/heatmap-batch?colormap=hot&alpha=0.6",
            files=files,
        )
        assert response.status_code == status.HTTP_200_OK

    def test_batch_heatmap_empty(self, test_app_gradcam):
        """Test empty batch heatmap."""
        response = test_app_gradcam.post("/api/inference/heatmap-batch", files=[])
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_heatmap_exceeds_limit(self, test_app_gradcam, sample_xray_bytes):
        """Test batch heatmap exceeding 500 limit."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_bytes), "image/png"))
            for i in range(501)
        ]
        response = test_app_gradcam.post("/api/inference/heatmap-batch", files=files)
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_batch_heatmap_single_image(self, test_app_gradcam, sample_xray_bytes):
        """Test batch heatmap with single image."""
        files = [("files", ("single.png", io.BytesIO(sample_xray_bytes), "image/png"))]
        response = test_app_gradcam.post("/api/inference/heatmap-batch", files=files)
        data = response.json()
        assert len(data["results"]) == 1
