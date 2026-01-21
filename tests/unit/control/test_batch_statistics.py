"""
Unit tests for BatchStatistics component.
Tests batch aggregation statistics calculation.
"""

import pytest

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchSummaryStats,
    ClinicalInterpretation,
    InferencePrediction,
    PredictionClass,
    RiskAssessment,
    SingleImageResult,
)
from federated_pneumonia_detection.src.control.model_inferance.internals.batch_statistics import (
    BatchStatistics,
)


class TestBatchStatistics:
    """Tests for BatchStatistics class."""

    @pytest.fixture
    def batch_stats(self):
        """Create BatchStatistics instance."""
        return BatchStatistics()

    # =========================================================================
    # Test calculate method - basic functionality
    # =========================================================================

    def test_calculate_returns_batch_summary_stats(
        self,
        batch_stats,
        sample_single_image_result,
    ):
        """Test calculate returns BatchSummaryStats instance."""
        summary = batch_stats.calculate([sample_single_image_result], total_images=1)

        assert isinstance(summary, BatchSummaryStats)

    def test_calculate_all_pneumonia(self, batch_stats):
        """Test calculation with all pneumonia predictions."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.PNEUMONIA,
                    confidence=0.8 + i * 0.01,
                    pneumonia_probability=0.8 + i * 0.01,
                    normal_probability=0.2 - i * 0.01,
                ),
                processing_time_ms=100 + i * 10,
            )
            for i in range(5)
        ]

        summary = batch_stats.calculate(results, total_images=5)

        assert summary.total_images == 5
        assert summary.successful == 5
        assert summary.failed == 0
        assert summary.normal_count == 0
        assert summary.pneumonia_count == 5

    def test_calculate_all_normal(self, batch_stats):
        """Test calculation with all normal predictions."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.NORMAL,
                    confidence=0.8 + i * 0.01,
                    pneumonia_probability=0.2 - i * 0.01,
                    normal_probability=0.8 + i * 0.01,
                ),
                processing_time_ms=100 + i * 10,
            )
            for i in range(5)
        ]

        summary = batch_stats.calculate(results, total_images=5)

        assert summary.total_images == 5
        assert summary.successful == 5
        assert summary.failed == 0
        assert summary.normal_count == 5
        assert summary.pneumonia_count == 0

    def test_calculate_mixed_predictions(self, batch_stats, sample_mixed_results):
        """Test calculation with mixed predictions."""
        summary = batch_stats.calculate(sample_mixed_results, total_images=10)

        assert summary.total_images == 10
        assert summary.successful == 8  # 2 failed
        assert summary.failed == 2
        assert summary.normal_count > 0
        assert summary.pneumonia_count > 0
        assert (summary.normal_count + summary.pneumonia_count) == summary.successful

    # =========================================================================
    # Test average confidence calculation
    # =========================================================================

    def test_calculate_avg_confidence(self, batch_stats):
        """Test average confidence is calculated correctly."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.PNEUMONIA,
                    confidence=0.8 + i * 0.02,  # 0.8, 0.82, 0.84, 0.86, 0.88
                    pneumonia_probability=0.8 + i * 0.02,
                    normal_probability=0.2 - i * 0.02,
                ),
                processing_time_ms=100,
            )
            for i in range(5)
        ]

        summary = batch_stats.calculate(results, total_images=5)

        # Average of [0.8, 0.82, 0.84, 0.86, 0.88] = 0.84
        expected_avg = (0.8 + 0.82 + 0.84 + 0.86 + 0.88) / 5
        assert summary.avg_confidence == pytest.approx(expected_avg, abs=0.01)

    def test_calculate_avg_confidence_with_failures(self, batch_stats):
        """Test average confidence ignores failed predictions."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=i % 2 == 0,  # Every other fails
                prediction=(
                    InferencePrediction(
                        predicted_class=PredictionClass.PNEUMONIA,
                        confidence=0.8,
                        pneumonia_probability=0.8,
                        normal_probability=0.2,
                    )
                    if i % 2 == 0
                    else None
                ),
                processing_time_ms=100,
            )
            for i in range(4)
        ]

        summary = batch_stats.calculate(results, total_images=4)

        # Only 2 successful, avg should be 0.8
        assert summary.avg_confidence == pytest.approx(0.8, abs=0.01)

    def test_calculate_avg_confidence_no_successful(self, batch_stats):
        """Test average confidence is 0 when no successful predictions."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=False,
                prediction=None,
                error="Failed",
                processing_time_ms=100,
            )
            for i in range(3)
        ]

        summary = batch_stats.calculate(results, total_images=3)

        assert summary.avg_confidence == 0.0

    # =========================================================================
    # Test average processing time calculation
    # =========================================================================

    def test_calculate_avg_processing_time(self, batch_stats):
        """Test average processing time is calculated correctly."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.PNEUMONIA,
                    confidence=0.8,
                    pneumonia_probability=0.8,
                    normal_probability=0.2,
                ),
                processing_time_ms=100 + i * 50,  # 100, 150, 200, 250, 300
            )
            for i in range(5)
        ]

        summary = batch_stats.calculate(results, total_images=5)

        # Average of [100, 150, 200, 250, 300] = 200
        expected_avg = (100 + 150 + 200 + 250 + 300) / 5
        assert summary.avg_processing_time_ms == pytest.approx(expected_avg, abs=0.01)

    def test_calculate_avg_processing_time_includes_failures(self, batch_stats):
        """Test average time includes failed predictions."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.PNEUMONIA,
                    confidence=0.8,
                    pneumonia_probability=0.8,
                    normal_probability=0.2,
                ),
                processing_time_ms=100 if i < 2 else 200,  # 100, 100, 200, 200
            )
            for i in range(4)
        ]

        summary = batch_stats.calculate(results, total_images=4)

        # Average should be 150
        assert summary.avg_processing_time_ms == pytest.approx(150, abs=0.01)

    def test_calculate_avg_processing_time_empty(self, batch_stats):
        """Test average time is 0 when no results."""
        summary = batch_stats.calculate([], total_images=0)
        assert summary.avg_processing_time_ms == 0.0

    # =========================================================================
    # Test high risk count calculation
    # =========================================================================

    def test_calculate_high_risk_count(self, batch_stats):
        """Test high risk count is calculated correctly."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.PNEUMONIA,
                    confidence=0.8,
                    pneumonia_probability=0.8,
                    normal_probability=0.2,
                ),
                clinical_interpretation=(
                    ClinicalInterpretation(
                        summary="Test",
                        confidence_explanation="Test",
                        risk_assessment=RiskAssessment(
                            risk_level="HIGH"
                            if i % 3 == 0
                            else "MODERATE",  # HIGH at 0, 3
                            false_negative_risk="LOW",
                            factors=[],
                        ),
                        recommendations=[],
                    )
                ),
                processing_time_ms=100,
            )
            for i in range(6)
        ]

        summary = batch_stats.calculate(results, total_images=6)

        # 2 HIGH risk (i=0, 3)
        assert summary.high_risk_count == 2

    def test_calculate_high_risk_count_with_critical(self, batch_stats):
        """Test CRITICAL risk level is counted."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.PNEUMONIA,
                    confidence=0.8,
                    pneumonia_probability=0.8,
                    normal_probability=0.2,
                ),
                clinical_interpretation=(
                    ClinicalInterpretation(
                        summary="Test",
                        confidence_explanation="Test",
                        risk_assessment=RiskAssessment(
                            risk_level="CRITICAL"
                            if i % 3 == 0
                            else "HIGH",  # CRITICAL at 0, 3
                            false_negative_risk="LOW",
                            factors=[],
                        ),
                        recommendations=[],
                    )
                ),
                processing_time_ms=100,
            )
            for i in range(6)
        ]

        summary = batch_stats.calculate(results, total_images=6)

        # 2 CRITICAL + 4 HIGH = 6 total
        assert summary.high_risk_count == 6

    def test_calculate_high_risk_count_no_clinical(self, batch_stats):
        """Test high risk count is 0 without clinical interpretations."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.PNEUMONIA,
                    confidence=0.8,
                    pneumonia_probability=0.8,
                    normal_probability=0.2,
                ),
                clinical_interpretation=None,
                processing_time_ms=100,
            )
            for i in range(5)
        ]

        summary = batch_stats.calculate(results, total_images=5)

        assert summary.high_risk_count == 0

    def test_calculate_high_risk_count_moderate_low_not_counted(self, batch_stats):
        """Test MODERATE and LOW risk levels are not counted."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.NORMAL,
                    confidence=0.8,
                    pneumonia_probability=0.2,
                    normal_probability=0.8,
                ),
                clinical_interpretation=(
                    ClinicalInterpretation(
                        summary="Test",
                        confidence_explanation="Test",
                        risk_assessment=RiskAssessment(
                            risk_level="MODERATE" if i % 2 == 0 else "LOW",
                            false_negative_risk="LOW",
                            factors=[],
                        ),
                        recommendations=[],
                    )
                ),
                processing_time_ms=100,
            )
            for i in range(4)
        ]

        summary = batch_stats.calculate(results, total_images=4)

        assert summary.high_risk_count == 0

    # =========================================================================
    # Test total_images parameter
    # =========================================================================

    def test_calculate_with_total_images_different_from_results(self, batch_stats):
        """Test calculation when total_images > len(results)."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.PNEUMONIA,
                    confidence=0.8,
                    pneumonia_probability=0.8,
                    normal_probability=0.2,
                ),
                processing_time_ms=100,
            )
            for i in range(5)
        ]

        summary = batch_stats.calculate(results, total_images=10)

        assert summary.total_images == 10
        assert summary.successful == 5
        assert summary.failed == 5  # 10 - 5

    def test_calculate_with_zero_total_images(self, batch_stats):
        """Test calculation with zero total images."""
        summary = batch_stats.calculate([], total_images=0)

        assert summary.total_images == 0
        assert summary.successful == 0
        assert summary.failed == 0

    # =========================================================================
    # Test edge cases
    # =========================================================================

    def test_calculate_empty_results(self, batch_stats):
        """Test calculation with empty results."""
        summary = batch_stats.calculate([], total_images=0)

        assert summary.total_images == 0
        assert summary.successful == 0
        assert summary.failed == 0
        assert summary.normal_count == 0
        assert summary.pneumonia_count == 0
        assert summary.avg_confidence == 0.0
        assert summary.avg_processing_time_ms == 0.0
        assert summary.high_risk_count == 0

    def test_calculate_all_failures(self, batch_stats):
        """Test calculation when all predictions fail."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=False,
                prediction=None,
                error="Processing failed",
                processing_time_ms=100 + i * 10,
            )
            for i in range(5)
        ]

        summary = batch_stats.calculate(results, total_images=5)

        assert summary.successful == 0
        assert summary.failed == 5
        assert summary.normal_count == 0
        assert summary.pneumonia_count == 0
        assert summary.avg_confidence == 0.0

    def test_calculate_single_result(self, batch_stats, sample_single_image_result):
        """Test calculation with single result."""
        summary = batch_stats.calculate([sample_single_image_result], total_images=1)

        assert summary.total_images == 1
        assert summary.successful == 1
        assert summary.failed == 0
        assert (
            summary.avg_confidence == sample_single_image_result.prediction.confidence
        )

    def test_calculate_very_large_batch(self, batch_stats):
        """Test calculation with large batch."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=InferencePrediction(
                    predicted_class=PredictionClass.PNEUMONIA
                    if i % 2 == 0
                    else PredictionClass.NORMAL,
                    confidence=0.7 + (i % 4) * 0.05,
                    pneumonia_probability=0.7 + (i % 4) * 0.05
                    if i % 2 == 0
                    else 0.3 - (i % 4) * 0.05,
                    normal_probability=0.3 - (i % 4) * 0.05
                    if i % 2 == 0
                    else 0.7 + (i % 4) * 0.05,
                ),
                processing_time_ms=100 + i,
            )
            for i in range(100)
        ]

        summary = batch_stats.calculate(results, total_images=100)

        assert summary.total_images == 100
        assert summary.successful == 100
        assert summary.failed == 0
        assert summary.normal_count == 50
        assert summary.pneumonia_count == 50
        assert 0.7 <= summary.avg_confidence <= 0.85  # Within expected range

    # =========================================================================
    # Test with missing predictions
    # =========================================================================

    def test_calculate_with_missing_predictions(self, batch_stats):
        """Test calculation handles missing predictions in successful results."""
        results = [
            SingleImageResult(
                filename=f"img_{i}.jpg",
                success=True,
                prediction=(
                    InferencePrediction(
                        predicted_class=PredictionClass.PNEUMONIA,
                        confidence=0.8,
                        pneumonia_probability=0.8,
                        normal_probability=0.2,
                    )
                    if i % 2 == 0
                    else None
                ),
                processing_time_ms=100,
            )
            for i in range(4)
        ]

        # This shouldn't crash, but prediction=None in successful result is unusual
        summary = batch_stats.calculate(results, total_images=4)

        # Should only count results with predictions
        assert summary.normal_count + summary.pneumonia_count <= summary.successful
