"""Weights & Biases tracker for inference endpoints.

Tracks prediction distributions, confidence scores, latency, and batch statistics
for model monitoring and drift detection.
"""

import logging
import os
from typing import Optional
from threading import Lock

import wandb

logger = logging.getLogger(__name__)


class WandbInferenceTracker:
    """Singleton tracker for logging inference metrics to W&B.

    Tracks:
    - Prediction distribution (pneumonia vs normal)
    - Confidence scores
    - Latency (ms)
    - Clinical interpretation usage
    - Error rates
    - Batch-level summaries
    """

    _instance: Optional["WandbInferenceTracker"] = None
    _lock = Lock()

    def __new__(cls) -> "WandbInferenceTracker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._run: Optional[wandb.sdk.wandb_run.Run] = None
        self._enabled = True

    def initialize(
        self,
        entity: str = "projectontheside25-multimedia-university",
        project: str = "FYP2",
        job_type: str = "inference",
        model_version: Optional[str] = None,
    ) -> bool:
        """Initialize W&B run for inference tracking.

        Args:
            entity: W&B team/entity name.
            project: W&B project name.
            job_type: Job type identifier.
            model_version: Optional model version tag.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        if self._run is not None:
            logger.debug("W&B run already initialized")
            return True

        try:
            self._run = wandb.init(
                entity=entity,
                project=project,
                job_type=job_type,
                config={
                    "model_version": model_version or "unknown",
                    "endpoint_type": "inference",
                },
                resume="allow",
            )
            logger.info(f"W&B inference tracking initialized: {self._run.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self._enabled = False
            return False

    def log_single_prediction(
        self,
        predicted_class: str,
        confidence: float,
        pneumonia_probability: float,
        normal_probability: float,
        processing_time_ms: float,
        clinical_interpretation_used: bool,
        model_version: str,
    ) -> None:
        """Log a single prediction to W&B.

        Args:
            predicted_class: "PNEUMONIA" or "NORMAL".
            confidence: Model confidence score (0-1).
            pneumonia_probability: Probability of pneumonia class.
            normal_probability: Probability of normal class.
            processing_time_ms: Inference latency in milliseconds.
            clinical_interpretation_used: Whether clinical agent was used.
            model_version: Model version string.
        """
        if not self._enabled or self._run is None:
            return

        try:
            self._run.log({
                # Prediction metrics
                "single/predicted_class": 1 if predicted_class == "PNEUMONIA" else 0,
                "single/confidence": confidence,
                "single/pneumonia_probability": pneumonia_probability,
                "single/normal_probability": normal_probability,

                # Performance metrics
                "single/latency_ms": processing_time_ms,

                # Service metrics
                "single/clinical_interpretation_used": int(clinical_interpretation_used),

                # Metadata
                "model_version": model_version,
            })
        except Exception as e:
            logger.warning(f"Failed to log single prediction to W&B: {e}")

    def log_batch_prediction(
        self,
        total_images: int,
        successful: int,
        failed: int,
        normal_count: int,
        pneumonia_count: int,
        avg_confidence: float,
        avg_processing_time_ms: float,
        total_processing_time_ms: float,
        high_risk_count: int,
        clinical_interpretation_used: bool,
        model_version: str,
    ) -> None:
        """Log batch prediction summary to W&B.

        Args:
            total_images: Total images in batch.
            successful: Number of successful predictions.
            failed: Number of failed predictions.
            normal_count: Number predicted as normal.
            pneumonia_count: Number predicted as pneumonia.
            avg_confidence: Average confidence across batch.
            avg_processing_time_ms: Average latency per image.
            total_processing_time_ms: Total batch processing time.
            high_risk_count: Number of high-risk predictions.
            clinical_interpretation_used: Whether clinical agent was used.
            model_version: Model version string.
        """
        if not self._enabled or self._run is None:
            return

        try:
            # Calculate derived metrics
            success_rate = successful / total_images if total_images > 0 else 0.0
            pneumonia_rate = pneumonia_count / successful if successful > 0 else 0.0
            high_risk_rate = high_risk_count / successful if successful > 0 else 0.0
            throughput = (successful / total_processing_time_ms * 1000) if total_processing_time_ms > 0 else 0.0

            self._run.log({
                # Batch composition
                "batch/total_images": total_images,
                "batch/successful": successful,
                "batch/failed": failed,
                "batch/success_rate": success_rate,

                # Prediction distribution
                "batch/normal_count": normal_count,
                "batch/pneumonia_count": pneumonia_count,
                "batch/pneumonia_rate": pneumonia_rate,

                # Risk metrics
                "batch/high_risk_count": high_risk_count,
                "batch/high_risk_rate": high_risk_rate,

                # Confidence metrics
                "batch/avg_confidence": avg_confidence,

                # Performance metrics
                "batch/avg_latency_ms": avg_processing_time_ms,
                "batch/total_time_ms": total_processing_time_ms,
                "batch/throughput_per_sec": throughput,

                # Service metrics
                "batch/clinical_interpretation_used": int(clinical_interpretation_used),

                # Metadata
                "model_version": model_version,
            })

            logger.debug(
                f"Logged batch: {total_images} images, "
                f"{pneumonia_rate:.1%} pneumonia, "
                f"{avg_processing_time_ms:.1f}ms avg latency"
            )
        except Exception as e:
            logger.warning(f"Failed to log batch prediction to W&B: {e}")

    def log_error(self, error_type: str, error_message: str) -> None:
        """Log an inference error to W&B.

        Args:
            error_type: Type of error (e.g., "validation", "model", "service").
            error_message: Error message.
        """
        if not self._enabled or self._run is None:
            return

        try:
            self._run.log({
                "errors/count": 1,
                "errors/type": error_type,
            })
        except Exception as e:
            logger.warning(f"Failed to log error to W&B: {e}")

    def finish(self) -> None:
        """Finish the W&B run."""
        if self._run is not None:
            try:
                self._run.finish()
                logger.info("W&B inference tracking finished")
            except Exception as e:
                logger.warning(f"Failed to finish W&B run: {e}")
            finally:
                self._run = None

    @property
    def is_active(self) -> bool:
        """Check if W&B tracking is active."""
        return self._enabled and self._run is not None


def get_wandb_tracker() -> WandbInferenceTracker:
    """Get the singleton W&B inference tracker instance."""
    return WandbInferenceTracker()
