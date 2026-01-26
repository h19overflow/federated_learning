import pytest
from fastapi import HTTPException
from federated_pneumonia_detection.src.api.deps import get_inference_service
from federated_pneumonia_detection.src.control.model_inferance.inference_service import (
    InferenceService,
)
from unittest.mock import MagicMock


def test_model_not_loaded(api_client_with_db, dummy_image_bytes):
    """
    Regression test: Ensure the API returns 503 Service Unavailable when the
    inference engine fails to load.
    """
    # Arrange
    # Create a mock service that simulates an unhealthy state
    unhealthy_service = MagicMock(spec=InferenceService)
    unhealthy_service.is_ready.return_value = False
    unhealthy_service.check_ready_or_raise.side_effect = HTTPException(
        status_code=503,
        detail="Inference model is not available. Please try again later.",
    )
    unhealthy_service.get_info.return_value = {
        "status": "unhealthy",
        "model_loaded": False,
        "gpu_available": False,
        "model_version": None,
    }

    # Override the dependency at the app level for this specific test
    from federated_pneumonia_detection.src.api.main import app

    app.dependency_overrides[get_inference_service] = lambda: unhealthy_service

    try:
        # Act
        # Attempt a prediction with a valid file type
        response = api_client_with_db.post(
            "/api/inference/predict",
            files={"file": ("test.jpg", dummy_image_bytes, "image/jpeg")},
        )

        # Assert
        assert response.status_code == 503
        assert "Inference model is not available" in response.json()["detail"]

        # Also check health endpoint
        health_response = api_client_with_db.get("/api/inference/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "unhealthy"
        assert health_response.json()["model_loaded"] is False

    finally:
        # Clean up the specific override
        if get_inference_service in app.dependency_overrides:
            del app.dependency_overrides[get_inference_service]
