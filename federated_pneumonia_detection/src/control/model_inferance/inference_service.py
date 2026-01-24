"""Unified inference service facade.

Composes all inference components and provides a clean interface for API consumption.
Import from here into api/deps.py for endpoint use.
"""

import logging
import time
from typing import Optional

from fastapi import HTTPException, UploadFile
from PIL import Image

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchInferenceResponse,
    InferencePrediction,
    InferenceResponse,
    PredictionClass,
    SingleImageResult,
)
from federated_pneumonia_detection.src.control.model_inferance.internals import (
    BatchStatistics,
    ImageProcessor,
    ImageValidator,
    InferenceEngine,
    ObservabilityLogger,
)

logger = logging.getLogger(__name__)


class InferenceService:
    """Unified inference service for API consumption.

    Composes all helper components and provides a clean interface.
    """

    def __init__(
        self,
        engine: Optional[InferenceEngine] = None,
    ):
        """Initialize with optional dependencies (lazy loading if None)."""
        self._engine = engine

        # Compose components
        self.validator = ImageValidator()
        self.processor = ImageProcessor()
        self.batch_stats = BatchStatistics()
        self.logger = ObservabilityLogger()

    @property
    def engine(self) -> Optional[InferenceEngine]:
        """Get inference engine (lazy loading)."""
        if self._engine is None:
            self._engine = _get_engine_singleton()
        return self._engine

    def is_ready(self) -> bool:
        """Check if service is ready for inference."""
        return self.engine is not None

    def check_ready_or_raise(self) -> None:
        """Raise HTTPException if service is not ready."""
        if not self.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Inference model is not available. Please try again later.",
            )

    def predict(self, image: Image.Image) -> tuple[str, float, float, float]:
        """Run inference on an image."""
        if self.engine is None:
            raise RuntimeError("Inference engine not available")
        return self.engine.predict(image)

    def create_prediction(
        self,
        predicted_class: str,
        confidence: float,
        pneumonia_prob: float,
        normal_prob: float,
    ) -> InferencePrediction:
        """Create prediction DTO from raw values."""
        return InferencePrediction(
            predicted_class=PredictionClass(predicted_class),
            confidence=confidence,
            pneumonia_probability=pneumonia_prob,
            normal_probability=normal_prob,
        )

    async def process_single(
        self,
        file: UploadFile,
    ) -> SingleImageResult:
        """Process a single image file end-to-end."""
        start_time = time.time()

        # Validate
        error = self.validator.validate(file)
        if error:
            return SingleImageResult(
                filename=file.filename or "unknown",
                success=False,
                error=error,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            # Read image
            image = await self.processor.read_from_upload(file)

            # Predict
            predicted_class, confidence, pneumonia_prob, normal_prob = self.predict(
                image,
            )
            prediction = self.create_prediction(
                predicted_class,
                confidence,
                pneumonia_prob,
                normal_prob,
            )

            return SingleImageResult(
                filename=file.filename or "unknown",
                success=True,
                prediction=prediction,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return SingleImageResult(
                filename=file.filename or "unknown",
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def predict_single(
        self,
        file: UploadFile,
    ) -> InferenceResponse:
        """Run inference on a single image.

        Handles validation, image reading, and prediction. Returns timing and response data.

        Args:
            file: Uploaded image file

        Returns:
            InferenceResponse object with prediction results
        """
        start_time = time.time()

        self.validator.validate_or_raise(file)
        self.check_ready_or_raise()

        image = await self.processor.read_from_upload(file)
        logger.info(
            f"Processing image: {file.filename}, size: {image.size}, mode: {image.mode}",
        )

        try:
            predicted_class, confidence, pneumonia_prob, normal_prob = self.predict(
                image,
            )
            prediction = self.create_prediction(
                predicted_class,
                confidence,
                pneumonia_prob,
                normal_prob,
            )

            processing_time_ms = (time.time() - start_time) * 1000
            model_version = self.engine.model_version if self.engine else "unknown"

            self.logger.log_single(
                predicted_class=predicted_class,
                confidence=confidence,
                pneumonia_prob=pneumonia_prob,
                normal_prob=normal_prob,
                processing_time_ms=processing_time_ms,
                clinical_used=False,
                model_version=model_version,
            )

            return InferenceResponse(
                success=True,
                prediction=prediction,
                model_version=model_version,
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            self.logger.log_error("inference", str(e))
            raise

    async def predict_batch(
        self,
        files: list,
    ) -> BatchInferenceResponse:
        """Run inference on multiple images with aggregated results.

        Processes images sequentially and returns aggregated results with
        summary statistics.

        Args:
            files: List of uploaded image files

        Returns:
            BatchInferenceResponse object with results and summary statistics
        """
        batch_start_time = time.time()

        self.check_ready_or_raise()

        max_batch_size = 500
        if len(files) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {max_batch_size} images allowed per batch request.",
            )

        results = []
        for file in files:
            result = await self.process_single(file=file)
            results.append(result)
            logger.info(f"Batch processing: {file.filename}, success: {result.success}")

        summary = self.batch_stats.calculate(results=results, total_images=len(files))

        total_batch_time_ms = (time.time() - batch_start_time) * 1000
        model_version = self.engine.model_version if self.engine else "unknown"

        self.logger.log_batch(
            summary=summary,
            total_time_ms=total_batch_time_ms,
            clinical_used=False,
            model_version=model_version,
        )

        return BatchInferenceResponse(
            success=True,
            results=results,
            summary=summary,
            model_version=model_version,
            total_processing_time_ms=total_batch_time_ms,
        )

    def get_info(self) -> dict:
        """Get service health info."""
        if self.engine is None:
            return {
                "status": "unhealthy",
                "model_loaded": False,
                "gpu_available": False,
                "model_version": None,
            }
        info = self.engine.get_info()
        return {
            "status": "healthy",
            "model_loaded": True,
            "gpu_available": info.get("gpu_available", False),
            "model_version": info.get("model_version"),
        }


# =============================================================================
# Singleton Management
# =============================================================================

_engine_instance: Optional[InferenceEngine] = None
_service_instance: Optional[InferenceService] = None


def _get_engine_singleton() -> Optional[InferenceEngine]:
    """Get or create InferenceEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        try:
            _engine_instance = InferenceEngine()
            logger.info("InferenceEngine initialized successfully")
        except Exception as e:
            logger.error(f"InferenceEngine initialization failed: {e}", exc_info=True)
            return None
    return _engine_instance


def get_inference_service() -> InferenceService:
    """Get InferenceService singleton for dependency injection."""
    global _service_instance
    if _service_instance is None:
        _service_instance = InferenceService()
    return _service_instance


def get_inference_engine() -> Optional[InferenceEngine]:
    """Get InferenceEngine singleton."""
    return _get_engine_singleton()
