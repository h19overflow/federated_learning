"""Inference service - abstraction layer for inference operations.

Provides a clean interface between the API layer and core logic.
Handles singleton management, error handling, and type conversions.
"""

import logging
from typing import Optional, Tuple

from PIL import Image

from federated_pneumonia_detection.src.control.model_inferance import InferenceEngine
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.clinical import (
    ClinicalInterpretationAgent,
)
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.clinical.clinical_agent import (
    ClinicalAnalysisResponse,
)

logger = logging.getLogger(__name__)

# Singleton instances
_inference_engine: Optional[InferenceEngine] = None
_clinical_agent: Optional[ClinicalInterpretationAgent] = None


class InferenceService:
    """Service layer for inference operations.

    Provides a unified interface for:
    - Model inference
    - Clinical interpretation
    - Health checks
    """

    def __init__(
        self,
        engine: Optional[InferenceEngine] = None,
        clinical_agent: Optional[ClinicalInterpretationAgent] = None,
    ):
        """Initialize service with optional injected dependencies.

        Args:
            engine: Optional InferenceEngine instance.
            clinical_agent: Optional ClinicalInterpretationAgent instance.
        """
        self._engine = engine
        self._clinical_agent = clinical_agent

    @property
    def engine(self) -> Optional[InferenceEngine]:
        """Get the inference engine (lazy loading)."""
        if self._engine is None:
            self._engine = get_inference_engine()
        return self._engine

    @property
    def clinical_agent(self) -> Optional[ClinicalInterpretationAgent]:
        """Get the clinical agent (lazy loading)."""
        if self._clinical_agent is None:
            self._clinical_agent = get_clinical_agent()
        return self._clinical_agent

    def predict(self, image: Image.Image) -> Tuple[str, float, float, float]:
        """Run inference on an image.

        Args:
            image: PIL Image to classify.

        Returns:
            Tuple of (predicted_class, confidence, pneumonia_prob, normal_prob).

        Raises:
            RuntimeError: If engine is not available.
        """
        if self.engine is None:
            raise RuntimeError("Inference engine not available")
        return self.engine.predict(image)

    async def get_clinical_interpretation(
        self,
        predicted_class: str,
        confidence: float,
        pneumonia_probability: float,
        normal_probability: float,
        image_info: Optional[dict] = None,
    ) -> Optional[ClinicalAnalysisResponse]:
        """Get clinical interpretation for a prediction.

        Args:
            predicted_class: NORMAL or PNEUMONIA
            confidence: Model confidence
            pneumonia_probability: Probability of pneumonia
            normal_probability: Probability of normal
            image_info: Optional image metadata

        Returns:
            ClinicalAnalysisResponse or None if unavailable.
        """
        if self.clinical_agent is None:
            return None

        return await self.clinical_agent.interpret(
            predicted_class=predicted_class,
            confidence=confidence,
            pneumonia_probability=pneumonia_probability,
            normal_probability=normal_probability,
            image_info=image_info,
        )

    def is_ready(self) -> bool:
        """Check if the service is ready for inference."""
        return self.engine is not None

    def get_health_info(self) -> dict:
        """Get health information about the service."""
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


def get_inference_engine() -> Optional[InferenceEngine]:
    """Get or create InferenceEngine singleton.

    Returns None if initialization fails (graceful degradation).
    """
    global _inference_engine

    if _inference_engine is None:
        try:
            _inference_engine = InferenceEngine()
            logger.info("InferenceEngine initialized successfully")
        except Exception as e:
            logger.error(f"InferenceEngine initialization failed: {e}", exc_info=True)
            return None

    return _inference_engine


def get_clinical_agent() -> Optional[ClinicalInterpretationAgent]:
    """Get or create ClinicalInterpretationAgent singleton.

    Returns None if initialization fails (graceful degradation).
    """
    global _clinical_agent

    if _clinical_agent is None:
        try:
            _clinical_agent = ClinicalInterpretationAgent()
            logger.info("ClinicalInterpretationAgent initialized successfully")
        except Exception as e:
            logger.warning(f"ClinicalInterpretationAgent unavailable: {e}")
            return None

    return _clinical_agent


def get_inference_service() -> InferenceService:
    """Get an InferenceService instance.

    The service handles lazy loading of its dependencies.
    """
    return InferenceService()


def is_model_loaded() -> bool:
    """Check if the inference model is loaded."""
    return _inference_engine is not None


def get_model_info() -> dict:
    """Get information about the loaded model."""
    if _inference_engine is None:
        return {
            "loaded": False,
            "version": None,
            "device": None,
        }
    return {
        "loaded": True,
        **_inference_engine.get_info(),
    }
