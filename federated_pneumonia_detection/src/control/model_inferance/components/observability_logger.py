"""Observability logging component for W&B metrics."""

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchSummaryStats,
)


class ObservabilityLogger:
    """Logs metrics to W&B."""

    def log_single(
        self,
        predicted_class: str,
        confidence: float,
        pneumonia_prob: float,
        normal_prob: float,
        processing_time_ms: float,
        clinical_used: bool,
        model_version: str,
    ) -> None:
        """Log single prediction metrics."""
        from federated_pneumonia_detection.src.control.dl_model.utils.data.wandb_inference_tracker import (
            get_wandb_tracker,
        )

        tracker = get_wandb_tracker()
        if tracker.is_active:
            tracker.log_single_prediction(
                predicted_class=predicted_class,
                confidence=confidence,
                pneumonia_probability=pneumonia_prob,
                normal_probability=normal_prob,
                processing_time_ms=processing_time_ms,
                clinical_interpretation_used=clinical_used,
                model_version=model_version,
            )

    def log_batch(
        self,
        summary: BatchSummaryStats,
        total_time_ms: float,
        clinical_used: bool,
        model_version: str,
    ) -> None:
        """Log batch prediction metrics."""
        from federated_pneumonia_detection.src.control.dl_model.utils.data.wandb_inference_tracker import (
            get_wandb_tracker,
        )

        tracker = get_wandb_tracker()
        if tracker.is_active:
            tracker.log_batch_prediction(
                total_images=summary.total_images,
                successful=summary.successful,
                failed=summary.failed,
                normal_count=summary.normal_count,
                pneumonia_count=summary.pneumonia_count,
                avg_confidence=summary.avg_confidence,
                avg_processing_time_ms=summary.avg_processing_time_ms,
                total_processing_time_ms=total_time_ms,
                high_risk_count=summary.high_risk_count,
                clinical_interpretation_used=clinical_used,
                model_version=model_version,
            )

    def log_error(self, error_type: str, message: str) -> None:
        """Log error to W&B."""
        from federated_pneumonia_detection.src.control.dl_model.utils.data.wandb_inference_tracker import (
            get_wandb_tracker,
        )

        tracker = get_wandb_tracker()
        if tracker.is_active:
            tracker.log_error(error_type, message)
