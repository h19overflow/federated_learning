"""
Unit tests for ObservabilityLogger component.
Tests logging functionality to W&B tracker.
"""

from unittest.mock import Mock, patch

import pytest

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchSummaryStats,
)
from federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger import (
    ObservabilityLogger,
)


class TestObservabilityLogger:
    """Tests for ObservabilityLogger class."""

    @pytest.fixture
    def logger(self):
        """Create ObservabilityLogger instance."""
        return ObservabilityLogger()

    @pytest.fixture
    def mock_tracker(self, mock_wandb_tracker):
        """Create mock W&B tracker."""
        return mock_wandb_tracker

    # =========================================================================
    # Test log_single method
    # =========================================================================

    def test_log_single_with_active_tracker(self, logger, mock_tracker):
        """Test logging single prediction with active tracker."""
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_single(
                predicted_class="PNEUMONIA",
                confidence=0.92,
                pneumonia_prob=0.92,
                normal_prob=0.08,
                processing_time_ms=150.5,
                clinical_used=True,
                model_version="v1.0",
            )

            # Should call tracker.log_single_prediction
            mock_tracker.log_single_prediction.assert_called_once()

            # Verify call arguments
            call_kwargs = mock_tracker.log_single_prediction.call_args[1]
            assert call_kwargs["predicted_class"] == "PNEUMONIA"
            assert call_kwargs["confidence"] == 0.92
            assert call_kwargs["pneumonia_probability"] == 0.92
            assert call_kwargs["normal_probability"] == 0.08
            assert call_kwargs["processing_time_ms"] == 150.5
            assert call_kwargs["clinical_interpretation_used"] is True
            assert call_kwargs["model_version"] == "v1.0"

    def test_log_single_with_inactive_tracker(self, logger):
        """Test logging with inactive tracker doesn't crash."""
        inactive_tracker = Mock()
        inactive_tracker.is_active = False

        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=inactive_tracker,
        ):
            # Should not raise
            logger.log_single(
                predicted_class="NORMAL",
                confidence=0.88,
                pneumonia_prob=0.12,
                normal_prob=0.88,
                processing_time_ms=120.3,
                clinical_used=False,
                model_version="v1.0",
            )

            # Should not call log_single_prediction
            inactive_tracker.log_single_prediction.assert_not_called()

    def test_log_single_normal_prediction(self, logger, mock_tracker):
        """Test logging NORMAL prediction."""
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_single(
                predicted_class="NORMAL",
                confidence=0.88,
                pneumonia_prob=0.12,
                normal_prob=0.88,
                processing_time_ms=100.0,
                clinical_used=False,
                model_version="v1.0",
            )

            mock_tracker.log_single_prediction.assert_called_once()

    def test_log_single_without_clinical(self, logger, mock_tracker):
        """Test logging without clinical interpretation."""
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_single(
                predicted_class="PNEUMONIA",
                confidence=0.75,
                pneumonia_prob=0.75,
                normal_prob=0.25,
                processing_time_ms=200.0,
                clinical_used=False,
                model_version="v1.1",
            )

            call_kwargs = mock_tracker.log_single_prediction.call_args[1]
            assert call_kwargs["clinical_interpretation_used"] is False

    def test_log_single_with_different_model_versions(self, logger, mock_tracker):
        """Test logging with various model versions."""
        versions = ["v1.0", "v2.0", "checkpoint_07", "model_0.928"]

        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            for version in versions:
                logger.log_single(
                    predicted_class="PNEUMONIA",
                    confidence=0.8,
                    pneumonia_prob=0.8,
                    normal_prob=0.2,
                    processing_time_ms=150.0,
                    clinical_used=True,
                    model_version=version,
                )

            # Should be called once per version
            assert mock_tracker.log_single_prediction.call_count == len(versions)

    # =========================================================================
    # Test log_batch method
    # =========================================================================

    def test_log_batch_with_active_tracker(
        self,
        logger,
        mock_tracker,
        sample_batch_summary,
    ):
        """Test logging batch statistics with active tracker."""
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_batch(
                summary=sample_batch_summary,
                total_time_ms=1500.5,
                clinical_used=True,
                model_version="v1.0",
            )

            mock_tracker.log_batch_prediction.assert_called_once()

            call_kwargs = mock_tracker.log_batch_prediction.call_args[1]
            assert call_kwargs["total_images"] == 10
            assert call_kwargs["successful"] == 9
            assert call_kwargs["failed"] == 1
            assert call_kwargs["normal_count"] == 4
            assert call_kwargs["pneumonia_count"] == 5
            assert call_kwargs["avg_confidence"] == 0.85
            assert call_kwargs["avg_processing_time_ms"] == 145.2
            assert call_kwargs["total_processing_time_ms"] == 1500.5
            assert call_kwargs["high_risk_count"] == 3
            assert call_kwargs["clinical_interpretation_used"] is True
            assert call_kwargs["model_version"] == "v1.0"

    def test_log_batch_with_inactive_tracker(self, logger, sample_batch_summary):
        """Test batch logging with inactive tracker."""
        inactive_tracker = Mock()
        inactive_tracker.is_active = False

        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=inactive_tracker,
        ):
            logger.log_batch(
                summary=sample_batch_summary,
                total_time_ms=1500.5,
                clinical_used=False,
                model_version="v1.0",
            )

            inactive_tracker.log_batch_prediction.assert_not_called()

    def test_log_batch_with_all_failures(self, logger, mock_tracker):
        """Test logging batch with all failures."""
        summary = BatchSummaryStats(
            total_images=5,
            successful=0,
            failed=5,
            normal_count=0,
            pneumonia_count=0,
            avg_confidence=0.0,
            avg_processing_time_ms=120.0,
            high_risk_count=0,
        )

        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_batch(
                summary=summary,
                total_time_ms=600.0,
                clinical_used=False,
                model_version="v1.0",
            )

            mock_tracker.log_batch_prediction.assert_called_once()

    def test_log_batch_with_all_success(self, logger, mock_tracker):
        """Test logging batch with all successes."""
        summary = BatchSummaryStats(
            total_images=10,
            successful=10,
            failed=0,
            normal_count=5,
            pneumonia_count=5,
            avg_confidence=0.88,
            avg_processing_time_ms=130.0,
            high_risk_count=2,
        )

        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_batch(
                summary=summary,
                total_time_ms=1300.0,
                clinical_used=True,
                model_version="v1.0",
            )

            mock_tracker.log_batch_prediction.assert_called_once()

    def test_log_batch_without_clinical(
        self,
        logger,
        mock_tracker,
        sample_batch_summary,
    ):
        """Test batch logging without clinical."""
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_batch(
                summary=sample_batch_summary,
                total_time_ms=1500.0,
                clinical_used=False,
                model_version="v1.1",
            )

            call_kwargs = mock_tracker.log_batch_prediction.call_args[1]
            assert call_kwargs["clinical_interpretation_used"] is False

    # =========================================================================
    # Test log_error method
    # =========================================================================

    def test_log_error_with_active_tracker(self, logger, mock_tracker):
        """Test error logging with active tracker."""
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_error(
                error_type="ValidationError",
                message="Invalid image format",
            )

            mock_tracker.log_error.assert_called_once_with(
                "ValidationError",
                "Invalid image format",
            )

    def test_log_error_with_inactive_tracker(self, logger):
        """Test error logging with inactive tracker."""
        inactive_tracker = Mock()
        inactive_tracker.is_active = False

        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=inactive_tracker,
        ):
            logger.log_error(
                error_type="ProcessingError",
                message="Failed to process image",
            )

            inactive_tracker.log_error.assert_not_called()

    def test_log_error_various_types(self, logger, mock_tracker):
        """Test logging various error types."""
        errors = [
            ("ValidationError", "Invalid image format"),
            ("ProcessingError", "Failed to load model"),
            ("TimeoutError", "Inference timed out"),
            ("ModelLoadError", "Checkpoint not found"),
        ]

        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            for error_type, message in errors:
                logger.log_error(error_type, message)

            assert mock_tracker.log_error.call_count == len(errors)

    # =========================================================================
    # Test edge cases
    # =========================================================================

    def test_log_single_with_extreme_values(self, logger, mock_tracker):
        """Test logging with extreme confidence values."""
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            # Very high confidence
            logger.log_single(
                predicted_class="PNEUMONIA",
                confidence=0.99,
                pneumonia_prob=0.99,
                normal_prob=0.01,
                processing_time_ms=1000.0,
                clinical_used=True,
                model_version="v1.0",
            )

            # Very low confidence
            logger.log_single(
                predicted_class="NORMAL",
                confidence=0.51,
                pneumonia_prob=0.49,
                normal_prob=0.51,
                processing_time_ms=50.0,
                clinical_used=False,
                model_version="v1.0",
            )

            assert mock_tracker.log_single_prediction.call_count == 2

    def test_log_single_with_zero_processing_time(self, logger, mock_tracker):
        """Test logging with zero or negative processing time."""
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_single(
                predicted_class="PNEUMONIA",
                confidence=0.8,
                pneumonia_prob=0.8,
                normal_prob=0.2,
                processing_time_ms=0.0,
                clinical_used=True,
                model_version="v1.0",
            )

            mock_tracker.log_single_prediction.assert_called_once()

    def test_log_batch_empty_batch(self, logger, mock_tracker):
        """Test logging empty batch."""
        summary = BatchSummaryStats(
            total_images=0,
            successful=0,
            failed=0,
            normal_count=0,
            pneumonia_count=0,
            avg_confidence=0.0,
            avg_processing_time_ms=0.0,
            high_risk_count=0,
        )

        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_batch(
                summary=summary,
                total_time_ms=0.0,
                clinical_used=False,
                model_version="v1.0",
            )

            mock_tracker.log_batch_prediction.assert_called_once()

    def test_log_error_empty_messages(self, logger, mock_tracker):
        """Test logging error with empty strings."""
        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_error("", "")

            mock_tracker.log_error.assert_called_once_with("", "")

    def test_log_error_long_messages(self, logger, mock_tracker):
        """Test logging error with very long messages."""
        long_message = "A" * 1000

        with patch(
            "federated_pneumonia_detection.src.control.model_inferance.internals.observability_logger.get_wandb_tracker",
            return_value=mock_tracker,
        ):
            logger.log_error("LongError", long_message)

            mock_tracker.log_error.assert_called_once()
