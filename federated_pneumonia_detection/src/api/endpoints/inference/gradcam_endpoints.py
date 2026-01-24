"""GradCAM heatmap generation endpoints.

Provides endpoints for generating GradCAM visualizations that highlight
the regions of chest X-rays that contributed most to the model's predictions.
"""

import time
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from federated_pneumonia_detection.src.api.deps import (
    get_inference_service,
    get_gradcam_service,
)
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchHeatmapResponse,
    HeatmapResponse,
)
from federated_pneumonia_detection.src.control.model_inferance import InferenceService
from federated_pneumonia_detection.src.control.model_inferance import GradCAMService

router = APIRouter(prefix="/api/inference", tags=["gradcam"])


@router.post("/heatmap", response_model=HeatmapResponse)
async def generate_heatmap(
    file: UploadFile = File(..., description="Chest X-ray image file (PNG, JPEG)"),
    colormap: str = Query(
        default="jet",
        description="Colormap for heatmap (jet, hot, viridis)",
    ),
    alpha: float = Query(
        default=0.4,
        ge=0.1,
        le=0.9,
        description="Heatmap overlay transparency (0.1-0.9)",
    ),
    service: InferenceService = Depends(get_inference_service),
    gradcam_service: GradCAMService = Depends(get_gradcam_service),
) -> HeatmapResponse:
    """Generate GradCAM heatmap visualization for a chest X-ray."""
    service.check_ready_or_raise()
    service.validator.validate_or_raise(file)
    image = await service.processor.read_from_upload(file, convert_rgb=True)

    result = await gradcam_service.generate_single_heatmap(
        image=image,
        filename=file.filename or "unknown",
        colormap=colormap,
        alpha=alpha,
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    return HeatmapResponse(
        success=True,
        filename=result.filename,
        heatmap_base64=str(result.heatmap_base64),
        original_image_base64=str(result.original_image_base64),
        processing_time_ms=result.processing_time_ms,
    )


@router.post("/heatmap-batch", response_model=BatchHeatmapResponse)
async def generate_heatmaps_batch(
    files: List[UploadFile] = File(
        ...,
        description="Chest X-ray image files (PNG, JPEG)",
    ),
    colormap: str = Query(
        default="jet",
        description="Colormap for heatmap (jet, hot, viridis)",
    ),
    alpha: float = Query(
        default=0.4,
        ge=0.1,
        le=0.9,
        description="Heatmap overlay transparency (0.1-0.9)",
    ),
    service: InferenceService = Depends(get_inference_service),
    gradcam_service: GradCAMService = Depends(get_gradcam_service),
) -> BatchHeatmapResponse:
    """Generate GradCAM heatmaps for multiple chest X-rays."""
    batch_start_time = time.time()

    service.check_ready_or_raise()

    # Read and validate all images
    images = []
    for file in files:
        try:
            service.validator.validate_or_raise(file)
            image = await service.processor.read_from_upload(file, convert_rgb=True)
            images.append((image, file.filename or "unknown"))
        except HTTPException as e:
            images.append((None, file.filename or "unknown"))

    results = await gradcam_service.generate_batch_heatmaps(
        images=images,
        colormap=colormap,
        alpha=alpha,
    )

    total_processing_time_ms = (time.time() - batch_start_time) * 1000

    return BatchHeatmapResponse(
        success=True,
        results=results,
        total_processing_time_ms=total_processing_time_ms,
    )
