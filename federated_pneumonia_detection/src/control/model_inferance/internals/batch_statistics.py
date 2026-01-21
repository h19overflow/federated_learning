"""Batch statistics calculation component."""

from typing import List

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchSummaryStats,
    PredictionClass,
    SingleImageResult,
)


class BatchStatistics:
    """Calculates batch prediction statistics."""

    def calculate(
        self,
        results: List[SingleImageResult],
        total_images: int,
    ) -> BatchSummaryStats:
        """Calculate summary statistics for batch predictions."""
        successful = [r.prediction for r in results if r.success and r.prediction]
        successful_count = len(successful)
        failed_count = total_images - successful_count

        normal_count = sum(
            1 for p in successful if p.predicted_class == PredictionClass.NORMAL
        )
        pneumonia_count = sum(
            1 for p in successful if p.predicted_class == PredictionClass.PNEUMONIA
        )

        avg_confidence = (
            sum(p.confidence for p in successful) / successful_count
            if successful_count > 0
            else 0.0
        )

        total_time = sum(r.processing_time_ms for r in results)
        avg_time = total_time / len(results) if results else 0.0

        high_risk_count = sum(
            1
            for r in results
            if r.clinical_interpretation
            and r.clinical_interpretation.risk_assessment.risk_level
            in ["HIGH", "CRITICAL"]
        )

        return BatchSummaryStats(
            total_images=total_images,
            successful=successful_count,
            failed=failed_count,
            normal_count=normal_count,
            pneumonia_count=pneumonia_count,
            avg_confidence=avg_confidence,
            avg_processing_time_ms=avg_time,
            high_risk_count=high_risk_count,
        )
