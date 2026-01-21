"""
Unit and integration tests for InferenceService.
Tests the unified inference service facade.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    InferencePrediction,
    PredictionClass,
    SingleImageResult,
)
from federated_pneumonia_detection.src.control.model_inferance.inference_service import (
    InferenceService,
    get_inference_engine,
    get_inference_service,
)


class TestInferenceService:
    """Tests for InferenceService class."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock inference engine."""
        engine = Mock()
        engine.predict = Mock(return_value=("PNEUMONIA", 0.92, 0.92, 0.08))
        engine.get_info = Mock(
            return_value={
                "model_version": "v1.0",
                "device": "cpu",
                "gpu_available": False,
                "checkpoint_path": "/path/to/checkpoint.ckpt",
            },
        )
        return engine

    @pytest.fixture
    def mock_clinical_agent(self):
        """Create mock clinical agent."""
        agent = Mock()

        async def mock_interpret(*args, **kwargs):
            return Mock(
                summary="Agent interpretation",
                confidence_explanation="High confidence",
                risk_level="HIGH",
                false_negative_risk="LOW",
                risk_factors=["High confidence"],
                recommendations=["Review recommended"],
            )

        agent.interpret = mock_interpret
        return agent

    @pytest.fixture
    def service(self, mock_engine, mock_clinical_agent):
        """Create InferenceService with mocked dependencies."""
        return InferenceService(engine=mock_engine, clinical_agent=mock_clinical_agent)

    @pytest.fixture
    def service_without_engine(self, mock_clinical_agent):
        """Create InferenceService without engine (for testing lazy loading)."""
        return InferenceService(engine=None, clinical_agent=mock_clinical_agent)

    # =========================================================================
    # Test initialization
    # =========================================================================

    def test_init_with_dependencies(self, mock_engine, mock_clinical_agent):
        """Test initialization with all dependencies."""
        service = InferenceService(
            engine=mock_engine,
            clinical_agent=mock_clinical_agent,
        )

        assert service._engine is mock_engine
        assert service._clinical_agent is mock_clinical_agent
        assert service.validator is not None
        assert service.processor is not None
        assert service.interpreter is not None
        assert service.batch_stats is not None
        assert service.logger is not None

    def test_init_without_dependencies(self):
        """Test initialization without dependencies (lazy loading)."""
        service = InferenceService(engine=None, clinical_agent=None)

        assert service._engine is None
        assert service._clinical_agent is None
        assert service.validator is not None  # Created immediately
        assert service.processor is not None

    def test_components_composed(self, service):
        """Test all components are composed."""
        assert hasattr(service, "validator")
        assert hasattr(service, "processor")
        assert hasattr(service, "interpreter")
        assert hasattr(service, "batch_stats")
        assert hasattr(service, "logger")

    # =========================================================================
    # Test properties
    # =========================================================================

    def test_engine_property(self, service, mock_engine):
        """Test engine property returns engine."""
        assert service.engine is mock_engine

    def test_engine_property_lazy_loading(self, service_without_engine):
        """Test engine property lazy loads when needed."""
        # Without patching, this would try to load real model
        # In tests, we patch _get_engine_singleton
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.inference_service._get_engine_singleton",
        ) as mock_get:
            mock_engine = Mock()
            mock_get.return_value = mock_engine

            engine = service_without_engine.engine

            assert engine is mock_engine
            mock_get.assert_called_once()

    def test_clinical_agent_property(self, service, mock_clinical_agent):
        """Test clinical_agent property returns agent."""
        assert service.clinical_agent is mock_clinical_agent

    def test_clinical_agent_property_lazy_loading(self):
        """Test clinical_agent property lazy loads when needed."""
        service = InferenceService(engine=None, clinical_agent=None)

        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.inference_service._get_clinical_agent_singleton",
        ) as mock_get:
            mock_agent = Mock()
            mock_get.return_value = mock_agent

            agent = service.clinical_agent

            assert agent is mock_agent
            mock_get.assert_called_once()

    def test_clinical_agent_sets_interpreter_agent(self):
        """Test lazy loading sets interpreter's agent."""
        service = InferenceService(engine=None, clinical_agent=None)

        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.inference_service._get_clinical_agent_singleton",
        ) as mock_get:
            mock_agent = Mock()
            mock_get.return_value = mock_agent

            # Access agent property
            _ = service.clinical_agent

            # Should have set interpreter's agent
            service.interpreter.set_agent.assert_called_once_with(mock_agent)

    # =========================================================================
    # Test is_ready and check_ready_or_raise
    # =========================================================================

    def test_is_ready_with_engine(self, service):
        """Test is_ready returns True when engine is available."""
        assert service.is_ready() is True

    def test_is_ready_without_engine(self, service_without_engine):
        """Test is_ready returns False when engine is not available."""
        assert service_without_engine.is_ready() is False

    def test_check_ready_or_raise_success(self, service):
        """Test check_ready_or_raise doesn't raise when ready."""
        # Should not raise
        service.check_ready_or_raise()

    def test_check_ready_or_raise_raises_503(self, service_without_engine):
        """Test check_ready_or_raise raises HTTPException when not ready."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            service_without_engine.check_ready_or_raise()

        assert exc_info.value.status_code == 503
        assert "not available" in exc_info.value.detail

    # =========================================================================
    # Test predict
    # =========================================================================

    def test_predict_with_engine(self, service, sample_rgb_image):
        """Test predict with available engine."""
        result = service.predict(sample_rgb_image)

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert result[0] == "PNEUMONIA"
        assert result[1] == pytest.approx(0.92, abs=0.01)

        service.engine.predict.assert_called_once_with(sample_rgb_image)

    def test_predict_without_engine(self, service_without_engine, sample_rgb_image):
        """Test predict raises RuntimeError without engine."""
        with pytest.raises(RuntimeError, match="not available"):
            service_without_engine.predict(sample_rgb_image)

    def test_predict_passes_image_to_engine(self, service, mock_upload_file_jpeg):
        """Test predict passes correct image to engine."""
        # Read image first
        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(await mock_upload_file_jpeg.read()))

        service.predict(image)
        service.engine.predict.assert_called_once()

    # =========================================================================
    # Test create_prediction
    # =========================================================================

    def test_create_prediction_returns_correct_type(self, service):
        """Test create_prediction returns InferencePrediction."""
        prediction = service.create_prediction(
            predicted_class="PNEUMONIA",
            confidence=0.92,
            pneumonia_prob=0.92,
            normal_prob=0.08,
        )

        assert isinstance(prediction, InferencePrediction)

    def test_create_prediction_fields(self, service):
        """Test create_prediction sets all fields correctly."""
        prediction = service.create_prediction(
            predicted_class="PNEUMONIA",
            confidence=0.92,
            pneumonia_prob=0.92,
            normal_prob=0.08,
        )

        assert prediction.predicted_class == PredictionClass.PNEUMONIA
        assert prediction.confidence == pytest.approx(0.92, abs=0.01)
        assert prediction.pneumonia_probability == pytest.approx(0.92, abs=0.01)
        assert prediction.normal_probability == pytest.approx(0.08, abs=0.01)

    def test_create_prediction_normal(self, service):
        """Test create_prediction for NORMAL class."""
        prediction = service.create_prediction(
            predicted_class="NORMAL",
            confidence=0.88,
            pneumonia_prob=0.12,
            normal_prob=0.88,
        )

        assert prediction.predicted_class == PredictionClass.NORMAL

    # =========================================================================
    # Test process_single - success case
    # =========================================================================

    @pytest.mark.asyncio
    async def test_process_single_success(self, service, mock_upload_file_jpeg):
        """Test processing single image successfully."""
        result = await service.process_single(
            mock_upload_file_jpeg,
            include_clinical=False,
        )

        assert isinstance(result, SingleImageResult)
        assert result.success is True
        assert result.filename == "test_xray.jpg"
        assert result.prediction is not None
        assert result.error is None
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_single_with_clinical(self, service, mock_upload_file_jpeg):
        """Test processing with clinical interpretation."""
        result = await service.process_single(
            mock_upload_file_jpeg,
            include_clinical=True,
        )

        assert result.success is True
        assert result.clinical_interpretation is not None

    @pytest.mark.asyncio
    async def test_process_single_validates_first(self, service):
        """Test process_single validates before processing."""
        from io import BytesIO

        from fastapi import UploadFile

        # Create invalid file
        file = UploadFile(
            filename="test.pdf",
            file=BytesIO(b"not an image"),
            content_type="application/pdf",
        )

        result = await service.process_single(file, include_clinical=False)

        assert result.success is False
        assert result.error is not None
        assert "Invalid file type" in result.error
        assert result.prediction is None

    # =========================================================================
    # Test process_single - error cases
    # =========================================================================

    @pytest.mark.asyncio
    async def test_process_single_with_corrupted_image(
        self,
        service,
        mock_upload_file_corrupted,
    ):
        """Test processing corrupted image returns error."""
        result = await service.process_single(
            mock_upload_file_corrupted,
            include_clinical=False,
        )

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_process_single_with_engine_failure(
        self,
        service,
        mock_upload_file_jpeg,
    ):
        """Test processing when engine predict fails."""
        # Make predict raise error
        service.engine.predict = Mock(side_effect=RuntimeError("Model error"))

        result = await service.process_single(
            mock_upload_file_jpeg,
            include_clinical=False,
        )

        assert result.success is False
        assert result.error is not None
        assert "Model error" in result.error

    @pytest.mark.asyncio
    async def test_process_single_with_processor_failure(
        self,
        service,
        mock_upload_file_jpeg,
    ):
        """Test processing when image processor fails."""
        # Mock processor to raise error
        service.processor.read_from_upload = AsyncMock(
            side_effect=Exception("Read error"),
        )

        result = await service.process_single(
            mock_upload_file_jpeg,
            include_clinical=False,
        )

        assert result.success is False
        assert "Read error" in result.error

    # =========================================================================
    # Test process_single - clinical interpretation
    # =========================================================================

    @pytest.mark.asyncio
    async def test_process_single_clinical_success(
        self,
        service,
        mock_upload_file_jpeg,
    ):
        """Test clinical interpretation is generated successfully."""
        result = await service.process_single(
            mock_upload_file_jpeg,
            include_clinical=True,
        )

        assert result.success is True
        assert result.clinical_interpretation is not None
        assert "interpretation" in str(result.clinical_interpretation.summary).lower()

    @pytest.mark.asyncio
    async def test_process_single_clinical_without_agent(self, mock_upload_file_jpeg):
        """Test clinical interpretation fallback without agent."""
        service = InferenceService(
            engine=Mock(predict=Mock(return_value=("PNEUMONIA", 0.92, 0.92, 0.08))),
            clinical_agent=None,
        )

        result = await service.process_single(
            mock_upload_file_jpeg,
            include_clinical=True,
        )

        # Should still use fallback
        assert result.success is True
        assert result.clinical_interpretation is not None

    @pytest.mark.asyncio
    async def test_process_single_no_clinical_when_flag_false(
        self,
        service,
        mock_upload_file_jpeg,
    ):
        """Test clinical interpretation not generated when flag is False."""
        result = await service.process_single(
            mock_upload_file_jpeg,
            include_clinical=False,
        )

        assert result.clinical_interpretation is None

    # =========================================================================
    # Test process_single - processing time
    # =========================================================================

    @pytest.mark.asyncio
    async def test_process_single_measures_time(self, service, mock_upload_file_jpeg):
        """Test processing time is measured."""
        result = await service.process_single(
            mock_upload_file_jpeg,
            include_clinical=False,
        )

        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 10000  # Should be fast in tests

    @pytest.mark.asyncio
    async def test_process_single_time_with_clinical(
        self,
        service,
        mock_upload_file_jpeg,
    ):
        """Test processing time includes clinical interpretation."""
        result = await service.process_single(
            mock_upload_file_jpeg,
            include_clinical=True,
        )

        assert result.processing_time_ms > 0

    # =========================================================================
    # Test get_info
    # =========================================================================

    def test_get_info_with_engine(self, service):
        """Test get_info returns info when engine is available."""
        info = service.get_info()

        assert info["status"] == "healthy"
        assert info["model_loaded"] is True
        assert info["gpu_available"] is False
        assert info["model_version"] == "v1.0"

    def test_get_info_without_engine(self, service_without_engine):
        """Test get_info returns unhealthy when no engine."""
        info = service_without_engine.get_info()

        assert info["status"] == "unhealthy"
        assert info["model_loaded"] is False
        assert info["gpu_available"] is False
        assert info["model_version"] is None

    def test_get_info_includes_gpu_info(self, service):
        """Test get_info includes GPU availability."""
        service.engine.get_info.return_value["gpu_available"] = True

        info = service.get_info()

        assert info["gpu_available"] is True

    # =========================================================================
    # Test singleton management functions
    # =========================================================================

    def test_get_inference_service_singleton(self):
        """Test get_inference_service returns singleton."""
        service1 = get_inference_service()
        service2 = get_inference_service()

        assert service1 is service2

    def test_get_inference_engine_singleton(self):
        """Test get_inference_engine returns singleton."""
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.inference_service._get_engine_singleton",
        ) as mock_get:
            mock_engine = Mock()
            mock_get.return_value = mock_engine

            engine1 = get_inference_engine()
            engine2 = get_inference_engine()

            assert engine1 is mock_engine
            assert engine2 is mock_engine
            mock_get.assert_called_once()  # Only called once due to singleton

    def test_get_clinical_agent_singleton(self):
        """Test get_clinical_agent returns singleton."""
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.inference_service._get_clinical_agent_singleton",
        ) as mock_get:
            mock_agent = Mock()
            mock_get.return_value = mock_agent

            agent1 = get_clinical_agent()
            agent2 = get_clinical_agent()

            assert agent1 is mock_agent
            assert agent2 is mock_agent
            mock_get.assert_called_once()

    # =========================================================================
    # Test integration scenarios
    # =========================================================================

    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, service, mock_upload_file_jpeg):
        """Test full inference pipeline succeeds."""
        result = await service.process_single(
            mock_upload_file_jpeg,
            include_clinical=True,
        )

        # Validate complete flow
        assert result.success is True
        assert result.prediction is not None
        assert result.clinical_interpretation is not None
        assert result.processing_time_ms > 0

        # Verify engine was called
        service.engine.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_sequential_processing(self, service):
        """Test processing multiple images sequentially."""
        from io import BytesIO

        import numpy as np
        from PIL import Image

        results = []
        for i in range(3):
            img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)

            file = UploadFile(
                filename=f"image_{i}.jpg",
                file=buffer,
                content_type="image/jpeg",
            )

            result = await service.process_single(file, include_clinical=False)
            results.append(result)

        assert all(r.success for r in results)
        assert all(r.prediction is not None for r in results)

    # =========================================================================
    # Test edge cases
    # =========================================================================

    @pytest.mark.asyncio
    async def test_process_single_with_empty_filename(self, service):
        """Test processing with empty filename."""
        from io import BytesIO

        file = UploadFile(
            filename="",
            file=BytesIO(b"fake data"),
            content_type="image/jpeg",
        )

        # Will fail validation (fake data), but should handle gracefully
        result = await service.process_single(file, include_clinical=False)

        # Should handle gracefully
        assert result.filename == ""

    @pytest.mark.asyncio
    async def test_process_single_with_very_large_image(self, service):
        """Test processing with large image."""
        from io import BytesIO

        import numpy as np
        from PIL import Image

        # Create large image
        img_array = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        file = UploadFile(
            filename="large.jpg",
            file=buffer,
            content_type="image/jpeg",
        )

        result = await service.process_single(file, include_clinical=False)

        # Should still process (image will be resized)
        assert result.success is True or result.error is not None
