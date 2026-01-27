from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from federated_pneumonia_detection.src.api.deps import (
    get_inference_service,
    get_query_engine,
)
from federated_pneumonia_detection.src.api.main import app
from federated_pneumonia_detection.src.control.model_inferance.inference_service import (  # noqa: E501
    InferenceService,
)


@pytest.fixture
def dummy_image_bytes():
    """Create a valid 1x1 pixel JPEG image for testing."""
    img = Image.new("RGB", (1, 1), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_inference_engine():
    """Mock InferenceEngine for testing."""
    mock = MagicMock()
    mock.model_version = "mock-1.0.0"
    mock.get_info.return_value = {"gpu_available": False, "model_version": "mock-1.0.0"}
    # Default prediction: NORMAL, Confidence, Pneumonia Prob, Normal Prob
    mock.predict.return_value = ("NORMAL", 0.99, 0.01, 0.99)
    return mock


@pytest.fixture
def client(mock_inference_engine):
    """FastAPI TestClient with overridden dependencies."""
    # Inject mock engine into InferenceService
    service = InferenceService(engine=mock_inference_engine)

    def override_get_inference_service():
        return service

    def override_get_query_engine():
        # As requested by Sub-Ares instructions
        return MagicMock()

    # Apply overrides
    app.dependency_overrides[get_inference_service] = override_get_inference_service
    app.dependency_overrides[get_query_engine] = override_get_query_engine

    with TestClient(app) as test_client:
        yield test_client

    # Clean up overrides after test
    app.dependency_overrides.clear()
