"""Tests for single image prediction endpoint.

Tests the /api/inference/predict endpoint for individual chest X-ray
predictions with and without clinical interpretation.
"""

import io

from fastapi import status
from PIL import Image

from tests.api.inference.conftest import (
    assert_valid_inference_response,
    sample_xray_png_bytes,
)


class TestSinglePredictionEndpoint:
    """Tests for POST /api/inference/predict endpoint."""

    # ==================== Success Cases ====================

    def test_single_prediction_success(self, test_inference_client, mock_upload_file):
        """Test successful single image prediction."""
        response = test_inference_client.post(
            "/api/inference/predict",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK
        assert_valid_inference_response(response.json())

    def test_single_prediction_with_clinical_interpretation(
        self,
        test_inference_client,
        mock_inference_service,
        mock_upload_file,
    ):
        """Test prediction with clinical interpretation enabled."""
        # Enable clinical interpretation
        response = test_inference_client.post(
            "/api/inference/predict?include_clinical_interpretation=true",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify clinical interpretation is present
        assert "clinical_interpretation" in data
        assert data["clinical_interpretation"] is not None
        assert "summary" in data["clinical_interpretation"]
        assert "risk_assessment" in data["clinical_interpretation"]

    def test_single_prediction_without_clinical_interpretation(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test prediction with clinical interpretation disabled (default)."""
        response = test_inference_client.post(
            "/api/inference/predict?include_clinical_interpretation=false",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Clinical interpretation should be None or not present
        if "clinical_interpretation" in data:
            assert data["clinical_interpretation"] is None

    def test_single_prediction_png_format(self, test_inference_client):
        """Test prediction with PNG image format."""
        # Create PNG image
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = test_inference_client.post(
            "/api/inference/predict",
            files={"file": ("test.png", buffer, "image/png")},
        )

        assert response.status_code == status.HTTP_200_OK

    def test_single_prediction_jpeg_format(self, test_inference_client):
        """Test prediction with JPEG image format."""
        # Create JPEG image
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        response = test_inference_client.post(
            "/api/inference/predict",
            files={"file": ("test.jpg", buffer, "image/jpeg")},
        )

        assert response.status_code == status.HTTP_200_OK

    def test_single_prediction_returns_model_version(self, test_inference_client):
        """Test that response includes model version."""
        response = test_inference_client.post(
            "/api/inference/predict",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        data = response.json()
        assert "model_version" in data
        assert isinstance(data["model_version"], str)

    def test_single_prediction_returns_processing_time(self, test_inference_client):
        """Test that response includes processing time."""
        response = test_inference_client.post(
            "/api/inference/predict",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        data = response.json()
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], (int, float))
        assert data["processing_time_ms"] >= 0

    def test_single_prediction_confidence_scores_valid(self, test_inference_client):
        """Test that confidence scores are within valid range."""
        response = test_inference_client.post(
            "/api/inference/predict",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        data = response.json()
        pred = data["prediction"]

        assert 0 <= pred["confidence"] <= 1
        assert 0 <= pred["pneumonia_probability"] <= 1
        assert 0 <= pred["normal_probability"] <= 1
        # Probabilities should sum to approximately 1
        assert (
            abs(pred["pneumonia_probability"] + pred["normal_probability"] - 1.0) < 0.01
        )

    def test_single_prediction_predicted_class_valid(self, test_inference_client):
        """Test that predicted class is one of the valid values."""
        response = test_inference_client.post(
            "/api/inference/predict",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        data = response.json()
        pred = data["prediction"]

        assert pred["predicted_class"] in ["NORMAL", "PNEUMONIA"]

    def test_single_prediction_service_methods_called(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test that service methods are called correctly."""
        response = test_inference_client.post(
            "/api/inference/predict",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify service methods were called
        mock_inference_service.check_ready_or_raise.assert_called_once()
        mock_inference_service.validator.validate_or_raise.assert_called_once()
        mock_inference_service.predict.assert_called_once()
        mock_inference_service.create_prediction.assert_called_once()

    def test_single_prediction_logging_called(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test that observability logging is called."""
        response = test_inference_client.post(
            "/api/inference/predict",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify logging was called
        mock_inference_service.logger.log_single.assert_called_once()

    # ==================== Error Cases ====================

    def test_single_prediction_missing_file(self, test_inference_client):
        """Test prediction with missing file parameter."""
        response = test_inference_client.post("/api/inference/predict")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_single_prediction_invalid_format(self, test_inference_client):
        """Test prediction with invalid file format (e.g., text file)."""
        response = test_inference_client.post(
            "/api/inference/predict",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )

        # Should be rejected by validator
        assert response.status_code != status.HTTP_200_OK

    def test_single_prediction_empty_file(self, test_inference_client):
        """Test prediction with empty file."""
        response = test_inference_client.post(
            "/api/inference/predict",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )

        # Should be rejected
        assert response.status_code != status.HTTP_200_OK

    def test_single_prediction_model_unavailable(self, test_unavailable_client):
        """Test prediction when model is not loaded."""
        response = test_unavailable_client.post(
            "/api/inference/predict",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "not available" in response.json()["detail"].lower()

    def test_single_prediction_corrupted_image(self, test_inference_client):
        """Test prediction with corrupted image data."""
        # Create invalid PNG data
        invalid_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = test_inference_client.post(
            "/api/inference/predict",
            files={"file": ("corrupt.png", io.BytesIO(invalid_data), "image/png")},
        )

        # Should fail gracefully
        assert response.status_code != status.HTTP_200_OK

    def test_single_prediction_service_exception_handling(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test that service exceptions are handled gracefully."""
        # Make predict method raise an exception
        mock_inference_service.predict.side_effect = Exception("Inference failed")

        response = test_inference_client.post(
            "/api/inference/predict",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "inference failed" in response.json()["detail"].lower()

    # ==================== Edge Cases ====================

    def test_single_prediction_large_filename(self, test_inference_client):
        """Test prediction with very long filename."""
        long_filename = "x" * 200 + ".png"

        response = test_inference_client.post(
            "/api/inference/predict",
            files={
                "file": (long_filename, io.BytesIO(sample_xray_png_bytes), "image/png"),
            },
        )

        # Should still work or fail gracefully
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
        ]

    def test_single_prediction_special_characters_filename(self, test_inference_client):
        """Test prediction with special characters in filename."""
        filename = "test_x-ray_🩺_#1.png"

        response = test_inference_client.post(
            "/api/inference/predict",
            files={"file": (filename, io.BytesIO(sample_xray_png_bytes), "image/png")},
        )

        # Should handle gracefully
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
        ]

    def test_single_prediction_case_insensitive_extension(self, test_inference_client):
        """Test prediction with uppercase file extension."""
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = test_inference_client.post(
            "/api/inference/predict",
            files={"file": ("test.PNG", buffer, "image/png")},
        )

        assert response.status_code == status.HTTP_200_OK

    def test_single_prediction_no_filename(self, test_inference_client):
        """Test prediction when filename is None."""
        response = test_inference_client.post(
            "/api/inference/predict",
            files={"file": (None, io.BytesIO(sample_xray_png_bytes), "image/png")},
        )

        # Should handle None filename gracefully
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]

    # ==================== Integration Tests ====================

    def test_single_prediction_response_schema_validation(self, test_inference_client):
        """Test that response matches Pydantic schema."""
        from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
            InferenceResponse,
        )

        response = test_inference_client.post(
            "/api/inference/predict",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK

        # Should not raise ValidationError
        InferenceResponse(**response.json())

    def test_single_prediction_concurrent_requests(self, test_inference_client):
        """Test that endpoint handles concurrent requests."""
        import threading

        results = []
        errors = []

        def make_request():
            try:
                response = test_inference_client.post(
                    "/api/inference/predict",
                    files={
                        "file": (
                            "test_xray.png",
                            io.BytesIO(sample_xray_png_bytes),
                            "image/png",
                        ),
                    },
                )
                results.append(response.status_code)
            except Exception as e:
                errors.append(e)

        # Make 5 concurrent requests
        threads = [threading.Thread(target=make_request) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All requests should succeed
        assert len(errors) == 0
        assert all(code == status.HTTP_200_OK for code in results)
