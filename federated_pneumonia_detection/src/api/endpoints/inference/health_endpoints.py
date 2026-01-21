"""Health check endpoint for inference service status.

Provides endpoint for monitoring the health and availability
of the inference service, including model loading status and GPU availability.
"""

import logging

from fastapi import APIRouter, Depends

from federated_pneumonia_detection.src.api.deps import get_inference_service
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    HealthCheckResponse,
)
from federated_pneumonia_detection.src.control.model_inferance import InferenceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    service: InferenceService = Depends(get_inference_service),
) -> HealthCheckResponse:
    """Check the health status of the inference service.

    Returns model loading status and GPU availability.
    """
    info = service.get_info()
    return HealthCheckResponse(
        status=info["status"],
        model_loaded=info["model_loaded"],
        gpu_available=info.get("gpu_available", False),
        model_version=info.get("model_version"),
    )
