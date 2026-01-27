"""Batch prediction endpoint for pneumonia detection.

Provides endpoint for running inference on multiple chest X-ray images
with aggregated results and summary statistics.
"""

from typing import List

from fastapi import APIRouter, Depends, File, UploadFile

from federated_pneumonia_detection.src.api.deps import get_inference_service
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchInferenceResponse,
)
from federated_pneumonia_detection.src.control.model_inferance import InferenceService

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.post("/predict-batch", response_model=BatchInferenceResponse)
async def predict_batch(
    files: List[UploadFile] = File(
        ...,
        description="Multiple chest X-ray images (PNG, JPEG)",
    ),
    service: InferenceService = Depends(get_inference_service),
) -> BatchInferenceResponse:
    """Run pneumonia detection on multiple chest X-ray images.

    Processes images sequentially and returns aggregated results with summary statistics.  # noqa: E501
    """
    service.check_ready_or_raise()
    return await service.predict_batch(files=files)
