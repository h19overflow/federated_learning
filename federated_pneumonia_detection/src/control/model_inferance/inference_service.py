"""Unified inference service facade.

Composes all inference components and provides a clean interface for API consumption.
Import from here into api/deps.py for endpoint use.
"""

import logging
import time
from typing import Optional, Tuple

from fastapi import HTTPException, UploadFile
from PIL import Image

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    InferencePrediction,
    PredictionClass,
    SingleImageResult,
)
from federated_pneumonia_detection.src.control.model_inferance.internals import (
    BatchStatistics,
    ClinicalInterpreter,
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
        clinical_agent=None,
    ):
        """Initialize with optional dependencies (lazy loading if None)."""
        self._engine = engine
        self._clinical_agent = clinical_agent

        # Compose components
        self.validator = ImageValidator()
        self.processor = ImageProcessor()
        self.interpreter = ClinicalInterpreter(clinical_agent)
        self.batch_stats = BatchStatistics()
        self.logger = ObservabilityLogger()

    @property
    def engine(self) -> Optional[InferenceEngine]:
        """Get inference engine (lazy loading)."""
        if self._engine is None:
            self._engine = _get_engine_singleton()
        return self._engine

    @property
    def clinical_agent(self):
        """Get clinical agent (lazy loading)."""
        if self._clinical_agent is None:
            self._clinical_agent = _get_clinical_agent_singleton()
            self.interpreter.set_agent(self._clinical_agent)
        return self._clinical_agent

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

    def predict(self, image: Image.Image) -> Tuple[str, float, float, float]:
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
        include_clinical: bool = False,
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

            # Clinical interpretation
            clinical = None
            if include_clinical:
                clinical = await self.interpreter.generate(
                    predicted_class=predicted_class,
                    confidence=confidence,
                    pneumonia_prob=pneumonia_prob,
                    normal_prob=normal_prob,
                    prediction=prediction,
                    image_info={"filename": file.filename, "size": image.size},
                )

            return SingleImageResult(
                filename=file.filename or "unknown",
                success=True,
                prediction=prediction,
                clinical_interpretation=clinical,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return SingleImageResult(
                filename=file.filename or "unknown",
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
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
_clinical_agent_instance = None
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


def _get_clinical_agent_singleton():
    """Get or create ClinicalInterpretationAgent singleton."""
    global _clinical_agent_instance
    if _clinical_agent_instance is None:
        try:
            from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.clinical import (
                ClinicalInterpretationAgent,
            )

            _clinical_agent_instance = ClinicalInterpretationAgent()
            logger.info("ClinicalInterpretationAgent initialized successfully")
        except Exception as e:
            logger.warning(f"ClinicalInterpretationAgent unavailable: {e}")
            return None
    return _clinical_agent_instance


def get_inference_service() -> InferenceService:
    """Get InferenceService singleton for dependency injection."""
    global _service_instance
    if _service_instance is None:
        _service_instance = InferenceService()
    return _service_instance


def get_inference_engine() -> Optional[InferenceEngine]:
    """Get InferenceEngine singleton."""
    return _get_engine_singleton()


def get_clinical_agent():
    """Get ClinicalInterpretationAgent singleton."""
    return _get_clinical_agent_singleton()
