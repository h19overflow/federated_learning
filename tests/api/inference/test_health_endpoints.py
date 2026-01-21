"""Tests for health check endpoint.

Tests the health check endpoint that monitors inference service status,
model loading, and GPU availability.
"""

from fastapi import status


class TestHealthCheckEndpoint:
    """Tests for GET /api/inference/health endpoint."""

    def test_health_check_returns_200(self, test_inference_client):
        """Test that health check endpoint returns 200 OK."""
        response = test_inference_client.get("/api/inference/health")

        assert response.status_code == status.HTTP_200_OK

    def test_health_check_response_structure(self, test_inference_client):
        """Test that health check response has correct structure."""
        response = test_inference_client.get("/api/inference/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check required fields
        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert "model_version" in data

        # Check types
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["gpu_available"], bool)
        assert data["model_version"] is None or isinstance(data["model_version"], str)

    def test_health_check_with_loaded_model(self, test_inference_client):
        """Test health check when model is loaded and ready."""
        response = test_inference_client.get("/api/inference/health")

        data = response.json()
        assert data["model_loaded"] is True
        # Model version comes from actual mock configuration
        assert data["model_version"] is not None

    def test_health_check_without_model(self, test_unavailable_client):
        """Test health check when model is not loaded."""
        response = test_unavailable_client.get("/api/inference/health")

        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert data["model_version"] is None

    def test_health_check_gpu_available_false(self, test_inference_client):
        """Test health check reports no GPU when unavailable."""
        response = test_inference_client.get("/api/inference/health")

        data = response.json()
        assert data["gpu_available"] is False

    def test_health_check_response_schema_validation(self, test_inference_client):
        """Test health check response matches Pydantic schema."""
        from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
            HealthCheckResponse,
        )

        response = test_inference_client.get("/api/inference/health")

        # Should not raise ValidationError if schema is correct
        HealthCheckResponse(**response.json())

    def test_health_check_no_auth_required(self, test_inference_client):
        """Test that health check doesn't require authentication."""
        # Endpoint should be accessible without auth headers
        response = test_inference_client.get("/api/inference/health")

        assert response.status_code == status.HTTP_200_OK

    def test_health_check_concurrent_requests(self, test_inference_client):
        """Test health check handles concurrent requests."""
        import threading

        results = []
        errors = []

        def make_request():
            try:
                response = test_inference_client.get("/api/inference/health")
                results.append(response.status_code)
            except Exception as e:
                errors.append(e)

        # Make 10 concurrent requests
        threads = [threading.Thread(target=make_request) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All requests should succeed
        assert len(errors) == 0
        assert all(code == status.HTTP_200_OK for code in results)
