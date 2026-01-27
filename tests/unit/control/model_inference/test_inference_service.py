from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchInferenceResponse,
    InferenceResponse,
    PredictionClass,
)
from federated_pneumonia_detection.src.control.model_inferance.inference_service import (  # noqa: E501
    InferenceService,
)


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.predict.return_value = ("PNEUMONIA", 0.95, 0.95, 0.05)
    engine.model_version = "v1.0.0"
    engine.get_info.return_value = {"gpu_available": True, "model_version": "v1.0.0"}
    return engine


@pytest.fixture
def mock_validator():
    validator = MagicMock()
    validator.validate.return_value = None
    return validator


@pytest.fixture
def mock_processor():
    processor = MagicMock()
    processor.read_from_upload = AsyncMock(return_value=Image.new("RGB", (224, 224)))
    return processor


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_batch_stats():
    stats = MagicMock()
    # Return a valid dict that matches BatchSummaryStats schema
    stats.calculate.return_value = {
        "total_images": 3,
        "successful": 3,
        "failed": 0,
        "normal_count": 0,
        "pneumonia_count": 3,
        "avg_confidence": 0.95,
        "avg_processing_time_ms": 10.0,
        "high_risk_count": 0,
    }
    return stats


@pytest.fixture
def inference_service(
    mock_engine, mock_validator, mock_processor, mock_logger, mock_batch_stats
):
    with (
        patch(
            "federated_pneumonia_detection.src.control.model_inferance."
            "inference_service.ImageValidator",
            return_value=mock_validator,
        ),
        patch(
            "federated_pneumonia_detection.src.control.model_inferance."
            "inference_service.ImageProcessor",
            return_value=mock_processor,
        ),
        patch(
            "federated_pneumonia_detection.src.control.model_inferance."
            "inference_service.BatchStatistics",
            return_value=mock_batch_stats,
        ),
        patch(
            "federated_pneumonia_detection.src.control.model_inferance."
            "inference_service.ObservabilityLogger",
            return_value=mock_logger,
        ),
        patch(
            "federated_pneumonia_detection.src.control.model_inferance."
            "inference_service._get_engine_singleton",
            return_value=mock_engine,
        ),
    ):
        service = InferenceService(engine=mock_engine)
        return service


class TestInferenceService:
    @pytest.mark.asyncio
    async def test_initialization(self, inference_service, mock_engine):
        """Verify service initializes with mocked components."""
        assert inference_service.engine == mock_engine
        assert inference_service.is_ready() is True

    @pytest.mark.asyncio
    async def test_predict_single_happy_path(
        self, inference_service, mock_engine, mock_processor, mock_validator
    ):
        """Test happy path for single image prediction."""
        # Setup
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.jpg"

        # Execute
        response = await inference_service.predict_single(mock_file)

        # Assert
        assert isinstance(response, InferenceResponse)
        assert response.success is True
        assert response.prediction.predicted_class == PredictionClass.PNEUMONIA
        assert response.prediction.confidence == 0.95
        assert response.model_version == "v1.0.0"

        mock_validator.validate_or_raise.assert_called_once_with(mock_file)
        mock_processor.read_from_upload.assert_called_once_with(mock_file)
        mock_engine.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_single_validation_failure(
        self, inference_service, mock_validator
    ):
        """Test predict_single when validation fails."""
        # Setup
        mock_file = MagicMock(spec=UploadFile)
        mock_validator.validate_or_raise.side_effect = HTTPException(
            status_code=400, detail="Invalid image"
        )

        # Execute & Assert
        with pytest.raises(HTTPException) as excinfo:
            await inference_service.predict_single(mock_file)

        assert excinfo.value.status_code == 400
        assert "Invalid image" in excinfo.value.detail

    @pytest.mark.asyncio
    async def test_predict_single_engine_not_ready(self, inference_service):
        """Test predict_single when engine is None."""
        # Setup
        inference_service._engine = None
        mock_file = MagicMock(spec=UploadFile)

        # Execute & Assert
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance."
            "inference_service._get_engine_singleton",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as excinfo:
                await inference_service.predict_single(mock_file)

            assert excinfo.value.status_code == 503

    @pytest.mark.asyncio
    async def test_predict_batch_happy_path(
        self, inference_service, mock_engine, mock_batch_stats
    ):
        """Test batch prediction with multiple files."""
        # Setup
        mock_files = [MagicMock(spec=UploadFile) for _ in range(3)]
        for i, f in enumerate(mock_files):
            f.filename = f"test_{i}.jpg"

        # Execute
        response = await inference_service.predict_batch(mock_files)

        # Assert
        assert isinstance(response, BatchInferenceResponse)
        assert response.success is True
        assert len(response.results) == 3
        assert mock_engine.predict.call_count == 3
        mock_batch_stats.calculate.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_batch_empty_list(
        self, inference_service, mock_engine, mock_batch_stats
    ):
        """Test batch prediction with empty list."""
        # Setup
        mock_files = []
        mock_batch_stats.calculate.return_value = {
            "total_images": 0,
            "successful": 0,
            "failed": 0,
            "normal_count": 0,
            "pneumonia_count": 0,
            "avg_confidence": 0.0,
            "avg_processing_time_ms": 0.0,
            "high_risk_count": 0,
        }

        # Execute
        response = await inference_service.predict_batch(mock_files)

        # Assert
        assert response.success is True
        assert len(response.results) == 0
        assert mock_engine.predict.call_count == 0

    @pytest.mark.asyncio
    async def test_predict_batch_exceeds_limit(self, inference_service):
        """Test batch prediction exceeding max limit."""
        # Setup
        mock_files = [MagicMock(spec=UploadFile) for _ in range(501)]

        # Execute & Assert
        with pytest.raises(HTTPException) as excinfo:
            await inference_service.predict_batch(mock_files)

        assert excinfo.value.status_code == 400
        assert "Maximum 500 images allowed" in excinfo.value.detail

    def test_get_info_healthy(self, inference_service, mock_engine):
        """Test get_info when engine is healthy."""
        # Execute
        info = inference_service.get_info()

        # Assert
        assert info["status"] == "healthy"
        assert info["model_loaded"] is True
        assert info["model_version"] == "v1.0.0"

    def test_get_info_unhealthy(self, inference_service):
        """Test get_info when engine is None."""
        # Setup
        inference_service._engine = None

        # Execute
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance."
            "inference_service._get_engine_singleton",
            return_value=None,
        ):
            info = inference_service.get_info()

        # Assert
        assert info["status"] == "unhealthy"
        assert info["model_loaded"] is False
        assert info["model_version"] is None
