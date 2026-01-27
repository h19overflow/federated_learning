"""Single image prediction endpoint for pneumonia detection.

Provides endpoint for running inference on individual chest X-ray images
with optional AI-generated clinical interpretation.
"""

from fastapi import APIRouter, Depends, File, UploadFile

from federated_pneumonia_detection.src.api.deps import get_inference_service
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    InferenceResponse,
)
from federated_pneumonia_detection.src.control.model_inferance import InferenceService

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.post("/predict", response_model=InferenceResponse)
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image file (PNG, JPEG)"),
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    """Run pneumonia detection on an uploaded chest X-ray image.

    Accepts PNG or JPEG images. Returns prediction with confidence scores.
    """
    service.check_ready_or_raise()
    return await service.predict_single(file=file)
