"""Tests for GradCAM heatmap generation endpoints.

Tests the /api/inference/heatmap and /api/inference/heatmap-batch endpoints
for generating visual explanations of model predictions.
"""

import io

from fastapi import status
from PIL import Image

from tests.api.inference.conftest import (
    assert_valid_heatmap_response,
    sample_xray_png_bytes,
)


class TestHeatmapEndpoint:
    """Tests for POST /api/inference/heatmap endpoint."""

    # ==================== Success Cases ====================

    def test_heatmap_generation_success(self, test_inference_client):
        """Test successful GradCAM heatmap generation."""
        response = test_inference_client.post(
            "/api/inference/heatmap",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK
        assert_valid_heatmap_response(response.json())

    def test_heatmap_generation_default_params(self, test_inference_client):
        """Test heatmap with default parameters (jet colormap, alpha=0.4)."""
        response = test_inference_client.post(
            "/api/inference/heatmap",
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

        # Default params should work
        assert data["success"] is True
        assert "heatmap_base64" in data

    def test_heatmap_with_jet_colormap(self, test_inference_client):
        """Test heatmap with jet colormap."""
        response = test_inference_client.post(
            "/api/inference/heatmap?colormap=jet",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_with_hot_colormap(self, test_inference_client):
        """Test heatmap with hot colormap."""
        response = test_inference_client.post(
            "/api/inference/heatmap?colormap=hot",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_with_viridis_colormap(self, test_inference_client):
        """Test heatmap with viridis colormap."""
        response = test_inference_client.post(
            "/api/inference/heatmap?colormap=viridis",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_with_alpha_0_1(self, test_inference_client):
        """Test heatmap with minimum alpha (0.1)."""
        response = test_inference_client.post(
            "/api/inference/heatmap?alpha=0.1",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_with_alpha_0_9(self, test_inference_client):
        """Test heatmap with maximum alpha (0.9)."""
        response = test_inference_client.post(
            "/api/inference/heatmap?alpha=0.9",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_with_alpha_0_5(self, test_inference_client):
        """Test heatmap with medium alpha (0.5)."""
        response = test_inference_client.post(
            "/api/inference/heatmap?alpha=0.5",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_custom_colormap_and_alpha(self, test_inference_client):
        """Test heatmap with custom colormap and alpha."""
        response = test_inference_client.post(
            "/api/inference/heatmap?colormap=hot&alpha=0.6",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_returns_filename(self, test_inference_client):
        """Test that heatmap response includes filename."""
        response = test_inference_client.post(
            "/api/inference/heatmap",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        data = response.json()
        assert "filename" in data
        assert data["filename"] == "test_xray.png"

    def test_heatmap_returns_processing_time(self, test_inference_client):
        """Test that heatmap response includes processing time."""
        response = test_inference_client.post(
            "/api/inference/heatmap",
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

    def test_heatmap_base64_format(self, test_inference_client):
        """Test that heatmap is returned as base64-encoded PNG."""
        response = test_inference_client.post(
            "/api/inference/heatmap",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        data = response.json()
        assert "heatmap_base64" in data
        assert data["heatmap_base64"].startswith("data:image/png;base64,")

    def test_heatmap_original_image_base64(self, test_inference_client):
        """Test that original image is also returned as base64."""
        response = test_inference_client.post(
            "/api/inference/heatmap",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        data = response.json()
        assert "original_image_base64" in data
        assert data["original_image_base64"].startswith("data:image/png;base64,")

    def test_heatmap_service_methods_called(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test that service methods are called correctly."""
        response = test_inference_client.post(
            "/api/inference/heatmap",
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

    # ==================== Error Cases ====================

    def test_heatmap_missing_file(self, test_inference_client):
        """Test heatmap generation with missing file."""
        response = test_inference_client.post("/api/inference/heatmap")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_heatmap_invalid_format(self, test_inference_client):
        """Test heatmap with invalid file format."""
        response = test_inference_client.post(
            "/api/inference/heatmap",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )

        assert response.status_code != status.HTTP_200_OK

    def test_heatmap_empty_file(self, test_inference_client):
        """Test heatmap with empty file."""
        response = test_inference_client.post(
            "/api/inference/heatmap",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )

        assert response.status_code != status.HTTP_200_OK

    def test_heatmap_model_unavailable(self, test_unavailable_client):
        """Test heatmap when model is not loaded."""
        response = test_unavailable_client.post(
            "/api/inference/heatmap",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_heatmap_invalid_alpha_below_min(self, test_inference_client):
        """Test heatmap with alpha below minimum (0.1)."""
        response = test_inference_client.post(
            "/api/inference/heatmap?alpha=0.05",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        # FastAPI validation should reject
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_heatmap_invalid_alpha_above_max(self, test_inference_client):
        """Test heatmap with alpha above maximum (0.9)."""
        response = test_inference_client.post(
            "/api/inference/heatmap?alpha=1.0",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        # FastAPI validation should reject
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_heatmap_invalid_alpha_negative(self, test_inference_client):
        """Test heatmap with negative alpha."""
        response = test_inference_client.post(
            "/api/inference/heatmap?alpha=-0.5",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_heatmap_corrupted_image(self, test_inference_client):
        """Test heatmap with corrupted image data."""
        invalid_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        response = test_inference_client.post(
            "/api/inference/heatmap",
            files={"file": ("corrupt.png", io.BytesIO(invalid_data), "image/png")},
        )

        assert response.status_code != status.HTTP_200_OK

    def test_heatmap_generation_exception_handling(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test that heatmap generation exceptions are handled."""
        # Make engine.model None to trigger error
        mock_inference_service.engine.model = None

        response = test_inference_client.post(
            "/api/inference/heatmap",
            files={
                "file": (
                    "test_xray.png",
                    io.BytesIO(sample_xray_png_bytes),
                    "image/png",
                ),
            },
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "heatmap generation failed" in response.json()["detail"].lower()

    # ==================== Edge Cases ====================

    def test_heatcase_special_characters_filename(self, test_inference_client):
        """Test heatmap with special characters in filename."""
        filename = "x-ray_🩺_#1.png"

        response = test_inference_client.post(
            "/api/inference/heatmap",
            files={"file": (filename, io.BytesIO(sample_xray_png_bytes), "image/png")},
        )

        # Should handle gracefully
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST,
        ]

    def test_heatmap_rgb_image(self, test_inference_client):
        """Test heatmap with RGB image."""
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = test_inference_client.post(
            "/api/inference/heatmap",
            files={"file": ("rgb.png", buffer, "image/png")},
        )

        assert response.status_code == status.HTTP_200_OK

    def test_heatmap_grayscale_image(self, test_inference_client):
        """Test heatmap with grayscale image."""
        img = Image.new("L", (224, 224), color=128)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = test_inference_client.post(
            "/api/inference/heatmap",
            files={"file": ("gray.png", buffer, "image/png")},
        )

        # Should convert to RGB internally
        assert response.status_code == status.HTTP_200_OK

    # ==================== Integration Tests ====================

    def test_heatmap_response_schema_validation(self, test_inference_client):
        """Test that response matches Pydantic schema."""
        from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
            HeatmapResponse,
        )

        response = test_inference_client.post(
            "/api/inference/heatmap",
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
        HeatmapResponse(**response.json())


class TestBatchHeatmapEndpoint:
    """Tests for POST /api/inference/heatmap-batch endpoint."""

    # ==================== Success Cases ====================

    def test_batch_heatmap_success(self, test_inference_client):
        """Test successful batch heatmap generation."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(3)
        ]

        response = test_inference_client.post(
            "/api/inference/heatmap-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["success"] is True
        assert "results" in data
        assert "total_processing_time_ms" in data

    def test_batch_heatmap_all_success(self, test_inference_client):
        """Test batch heatmap where all images succeed."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(3)
        ]

        response = test_inference_client.post(
            "/api/inference/heatmap-batch",
            files=files,
        )

        data = response.json()
        assert len(data["results"]) == 3
        assert all(result["success"] for result in data["results"])

    def test_batch_heatmap_with_params(self, test_inference_client):
        """Test batch heatmap with custom parameters."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(2)
        ]

        response = test_inference_client.post(
            "/api/inference/heatmap-batch?colormap=hot&alpha=0.6",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK

    def test_batch_heatmap_mixed_formats(self, test_inference_client):
        """Test batch heatmap with PNG and JPEG images."""
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
            "/api/inference/heatmap-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["results"]) == 2

    def test_batch_heatmap_returns_total_time(self, test_inference_client):
        """Test that batch heatmap returns total processing time."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(3)
        ]

        response = test_inference_client.post(
            "/api/inference/heatmap-batch",
            files=files,
        )

        data = response.json()
        assert "total_processing_time_ms" in data
        assert data["total_processing_time_ms"] >= 0

    def test_batch_heatmap_service_methods_called(
        self,
        test_inference_client,
        mock_inference_service,
    ):
        """Test that service methods are called for batch heatmap."""
        files = [
            ("files", ("test.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_inference_client.post(
            "/api/inference/heatmap-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK

        # Verify service methods were called
        mock_inference_service.check_ready_or_raise.assert_called_once()

    # ==================== Error Cases ====================

    def test_batch_heatmap_empty_batch(self, test_inference_client):
        """Test batch heatmap with no files."""
        response = test_inference_client.post("/api/inference/heatmap-batch", files=[])

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_heatmap_exceeds_max_limit(self, test_inference_client):
        """Test batch heatmap exceeding 500 image limit."""
        files = [
            ("files", (f"xray_{i}.png", io.BytesIO(sample_xray_png_bytes), "image/png"))
            for i in range(501)
        ]

        response = test_inference_client.post(
            "/api/inference/heatmap-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_batch_heatmap_model_unavailable(self, test_unavailable_client):
        """Test batch heatmap when model is not loaded."""
        files = [
            ("files", ("test.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_unavailable_client.post(
            "/api/inference/heatmap-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_batch_heatmap_mixed_success_failure(self, test_inference_client):
        """Test batch heatmap with mixed successful and failed results."""
        # Create valid PNG
        valid_buffer = io.BytesIO(sample_xray_png_bytes)

        # Create invalid text file
        invalid_buffer = io.BytesIO(b"not an image")

        files = [
            ("files", ("valid.png", valid_buffer, "image/png")),
            ("files", ("invalid.txt", invalid_buffer, "text/plain")),
        ]

        response = test_inference_client.post(
            "/api/inference/heatmap-batch",
            files=files,
        )

        data = response.json()
        assert data["success"] is True  # Batch completed
        assert any(not result["success"] for result in data["results"])

    # ==================== Edge Cases ====================

    def test_batch_heatmap_single_image(self, test_inference_client):
        """Test batch heatmap with single image."""
        files = [
            ("files", ("single.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_inference_client.post(
            "/api/inference/heatmap-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["results"]) == 1

    def test_batch_heatmap_concurrent_requests(self, test_inference_client):
        """Test that batch heatmap handles concurrent requests."""
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
                    "/api/inference/heatmap-batch",
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

        assert len(errors) == 0
        assert all(code == status.HTTP_200_OK for code in results)

    # ==================== Integration Tests ====================

    def test_batch_heatmap_response_schema_validation(self, test_inference_client):
        """Test that batch heatmap response matches Pydantic schema."""
        from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
            BatchHeatmapResponse,
        )

        files = [
            ("files", ("test.png", io.BytesIO(sample_xray_png_bytes), "image/png")),
        ]

        response = test_inference_client.post(
            "/api/inference/heatmap-batch",
            files=files,
        )

        assert response.status_code == status.HTTP_200_OK

        # Should not raise ValidationError
        BatchHeatmapResponse(**response.json())
