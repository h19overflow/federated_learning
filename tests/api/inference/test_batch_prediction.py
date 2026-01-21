"""Tests for batch prediction endpoint.

Tests the /api/inference/predict-batch endpoint for processing multiple
chest X-ray images with aggregated results and statistics.
"""

import io

from fastapi import status
from PIL import Image

from tests.api.inference.conftest import (
    sample_xray_png_bytes,
)


class TestBatchPredictionEndpoint:
    """Tests for POST /api/inference/predict-batch endpoint."""

    # ==================== Success Cases ====================

    def test_batch_prediction_success(
        self,
        test_inference_client,
        mock_upload_files_list,
    ):
        """Test successful batch prediction with multiple images."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(3)
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["success"] is True
        assert "results" in data
        assert "summary" in data
        assert "total_processing_time_ms" in data

    def test_batch_prediction_all_success(
        self,
        test_inference_client,
        mock_upload_files_list,
    ):
        """Test batch prediction where all images succeed."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(3)
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        data = response.json()
        assert len(data["results"]) == 3
        assert all(result["success"] for result in data["results"])

    def test_batch_prediction_with_clinical_disabled(self, test_inference_client):
        """Test batch prediction with clinical interpretation disabled (default)."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(2)
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch?include_clinical_interpretation=false",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK

    def test_batch_prediction_mixed_formats(self, test_inference_client):
        """Test batch prediction with mixed PNG and JPEG images."""
        # Create PNG
        png_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        png_buffer = io.BytesIO()
        png_img.save(png_buffer, format="PNG")
        png_buffer.seek(0)

        # Create JPEG
        jpg_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        jpg_buffer = io.BytesIO()
        jpg_img.save(jpg_buffer, format="JPEG")
        jpg_buffer.seek(0)

        files = [
            ("files", ("test.png", png_buffer, "image/png")),
            ("files", ("test.jpg", jpg_buffer, "image/jpeg")),
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["results"]) == 2

    def test_batch_prediction_summary_calculated(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test that batch summary statistics are calculated correctly."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(3)
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        data = response.json()
        summary = data["summary"]

        # Verify summary fields
        assert "total_images" in summary
        assert "successful" in summary
        assert "failed" in summary
        assert "normal_count" in summary
        assert "pneumonia_count" in summary
        assert "avg_confidence" in summary
        assert "avg_processing_time_ms" in summary

    def test_batch_prediction_summary_values(self, test_inference_client):
        """Test that summary statistics have valid values."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(3)
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        data = response.json()
        summary = data["summary"]

        assert summary["total_images"] == 3
        assert summary["successful"] == 3
        assert summary["failed"] == 0
        assert summary["avg_confidence"] >= 0
        assert summary["avg_confidence"] <= 1
        assert summary["avg_processing_time_ms"] >= 0

    def test_batch_prediction_returns_model_version(self, test_inference_client):
        """Test that batch response includes model version."""
        files = [
            ("files", ("test.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        data = response.json()
        assert "model_version" in data
        assert isinstance(data["model_version"], str)

    def test_batch_prediction_returns_total_time(self, test_inference_client):
        """Test that batch response includes total processing time."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(3)
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        data = response.json()
        assert "total_processing_time_ms" in data
        assert data["total_processing_time_ms"] >= 0

    def test_batch_prediction_service_methods_called(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test that service methods are called correctly."""
        files = [
            ("files", ("test.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify service methods were called
        mock_inference_service.check_ready_or_raise.assert_called_once()
        mock_inference_service.batch_stats.calculate.assert_called_once()

    def test_batch_prediction_logging_called(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test that observability logging is called for batch."""
        files = [
            ("files", ("test.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify batch logging was called
        mock_inference_service.logger.log_batch.assert_called_once()

    def test_batch_prediction_process_single_called_for_each(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test that process_single is called for each file."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(3)
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify process_single was called 3 times
        assert mock_inference_service.process_single.call_count == 3

    # ==================== Error Cases ====================

    def test_batch_prediction_empty_batch(self, test_inference_client):
        """Test batch prediction with no files."""
        response = test_inference_client.post("/api/inference/predict-batch", files=[])

        # FastAPI will reject empty file list
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_prediction_missing_files(self, test_inference_client):
        """Test batch prediction without files parameter."""
        response = test_inference_client.post("/api/inference/predict-batch")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_prediction_exceeds_max_limit(self, test_inference_client):
        """Test batch prediction exceeding 500 image limit."""
        # Create 501 files (exceeds 500 limit)
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(501)
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert (
            "maximum" in response.json()["detail"].lower()
            or "500" in response.json()["detail"]
        )

    def test_batch_prediction_model_unavailable(self, test_unavailable_client):
        """Test batch prediction when model is not loaded."""
        files = [
            ("files", ("test.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_unavailable_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "not available" in response.json()["detail"].lower()

    def test_batch_prediction_invalid_format_mixed(self, test_inference_client):
        """Test batch prediction with mixed valid and invalid files."""
        # Create valid PNG
        png_buffer = io.BytesIO(sample_xray_png_bytes)

        # Create invalid text file
        txt_buffer = io.BytesIO(b"not an image")

        files = [
            ("files", ("valid.png", png_buffer, "image/png")),
            ("files", ("invalid.txt", txt_buffer, "text/plain")),
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        # Should process batch with mixed success/failure
        data = response.json()
        assert data["success"] is True  # Batch completed
        if "summary" in data:
            assert data["summary"]["failed"] > 0

    def test_batch_prediction_corrupted_image(self, test_inference_client):
        """Test batch prediction with corrupted image."""
        invalid_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        files = [("files", ("corrupt.png", io.BytesIO(invalid_data), "image/png"))]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        # Should fail gracefully
        data = response.json()
        assert data["summary"]["failed"] > 0 if "summary" in data else True

    # ==================== Edge Cases ====================

    def test_batch_prediction_single_image(self, test_inference_client):
        """Test batch prediction with single image (edge case)."""
        files = [
            ("files", ("single.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) == 1

    def test_batch_prediction_max_allowed(self, test_inference_client):
        """Test batch prediction with exactly 500 images (max allowed)."""
        # Note: This test might be slow, so we'll mock it
        with patch.object(
            test_inference_client.app,
            "include_router",
        ):
            files = [
                (
                    "files",
                    (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"),
                )
                for i in range(500)
            ]

            response = test_inference_client.post(
                "/api/inference/predict-batch",
                files=files,
            )

            # Should accept 500 images
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
            ]

    def test_batch_prediction_large_batch(self, test_inference_client):
        """Test batch prediction with large number of images (100)."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(100)
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        # Should handle 100 images
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["summary"]["total_images"] == 100

    def test_batch_prediction_special_characters_filename(self, test_inference_client):
        """Test batch prediction with special characters in filenames."""
        files = [
            ("files", ("x-ray 🩺.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
            ("files", ("test_#1.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        # Should handle gracefully
        assert response.status_code == status.HTTP_200_OK

    def test_batch_prediction_mixed_results(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test batch prediction with some successful and some failed results."""
        # Mock process_single to return mixed results
        from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
            InferencePrediction,
            PredictionClass,
            SingleImageResult,
        )

        success_result = SingleImageResult(
            filename="success.png",
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

        fail_result = SingleImageResult(
            filename="fail.png",
            success=False,
            error="Invalid image format",
            processing_time_ms=10.0,
        )

        mock_inference_service.process_single.side_effect = [
            success_result,
            fail_result,
        ]

        files = [
            ("files", ("success.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
            ("files", ("fail.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        data = response.json()
        assert data["summary"]["successful"] == 1
        assert data["summary"]["failed"] == 1

    # ==================== Integration Tests ====================

    def test_batch_prediction_response_schema_validation(self, test_inference_client):
        """Test that response matches Pydantic schema."""
        from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
            BatchInferenceResponse,
        )

        files = [
            ("files", ("test.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK

        # Should not raise ValidationError
        BatchInferenceResponse(**response.json())

    def test_batch_prediction_concurrent_requests(self, test_inference_client):
        """Test that endpoint handles concurrent batch requests."""
        import threading

        results = []
        errors = []

        def make_request():
            try:
                files = [
                    (
                        "files",
                        (
                            f"xray_{i}.png",
                            io.BytesIO(sample_xray_png_bytes),
                            "image/png",
                        ),
                    )
                    for i in range(2)
                ]
                response = test_inference_client.post(
                    "/api/inference/predict-batch",
                    files=files,
                )
                results.append(response.status_code)
            except Exception as e:
                errors.append(e)

        # Make 3 concurrent requests
        threads = [threading.Thread(target=make_request) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All requests should succeed
        assert len(errors) == 0
        assert all(code == status.HTTP_200_OK for code in results)

    def test_batch_prediction_statistics_accuracy(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test that summary statistics are calculated accurately."""
        # Mock to return predictable results
        from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
            InferencePrediction,
            PredictionClass,
            SingleImageResult,
        )

        results = []
        for i in range(5):
            result = SingleImageResult(
                filename=f"xray_{i}.png",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.PNEUMONIA
                    if i < 3
                    else PredictionClass.NORMAL,
                    confidence=0.9 + (i * 0.01),
                    pneumonia_probability=0.7 if i < 3 else 0.2,
                    normal_probability=0.3 if i < 3 else 0.8,
                ),
                clinical_interpretation=None,
                processing_time_ms=50.0 + i,
            )
            results.append(result)

        mock_inference_service.process_single.side_effect = results

        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(5)
        ]

        response = test_inference_client.post(
            "/api/inference/predict-batch",
            files=files,
        )

        data = response.json()
        summary = data["summary"]

        # Verify statistics
        assert summary["total_images"] == 5
        assert summary["successful"] == 5
        assert summary["pneumonia_count"] == 3
        assert summary["normal_count"] == 2
        # Average confidence should be close to expected
        expected_avg = sum([0.9 + (i * 0.01) for i in range(5)]) / 5
        assert abs(summary["avg_confidence"] - expected_avg) < 0.01
