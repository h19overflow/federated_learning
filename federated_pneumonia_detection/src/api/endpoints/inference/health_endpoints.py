"""Health check endpoint for inference service status.

Provides endpoint for monitoring the health and availability
of the inference service, including model loading status and GPU availability.
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends

from federated_pneumonia_detection.src.api.deps import get_inference_engine
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    HealthCheckResponse,
)
from federated_pneumonia_detection.src.control.model_inferance.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    engine: Optional[InferenceEngine] = Depends(get_inference_engine),
) -> HealthCheckResponse:
    """Check the health status of the inference service.

    Returns model loading status and GPU availability.
    """
    if engine is None:
        return HealthCheckResponse(
            status="unhealthy",
            model_loaded=False,
            gpu_available=False,
            model_version=None,
        )

    info = engine.get_info()
    return HealthCheckResponse(
        status="healthy",
        model_loaded=True,
        gpu_available=info.get("gpu_available", False),
        model_version=info.get("model_version"),
    )
