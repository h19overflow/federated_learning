"""GradCAM heatmap generation endpoints.

Provides endpoints for generating GradCAM visualizations that highlight
the regions of chest X-rays that contributed most to the model's predictions.
"""

import logging
import time
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from PIL import Image

from federated_pneumonia_detection.src.api.deps import get_inference_service
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchHeatmapItem,
    BatchHeatmapResponse,
    HeatmapResponse,
)
from federated_pneumonia_detection.src.control.model_inferance import InferenceService
from federated_pneumonia_detection.src.control.model_inferance.gradcam import (
    GradCAM,
    heatmap_to_base64,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["gradcam"])


def _generate_single_heatmap(
    image: Image.Image,
    filename: str,
    service: InferenceService,
    colormap: str = "jet",
    alpha: float = 0.4,
) -> BatchHeatmapItem:
    """Generate heatmap for a single image."""
    start_time = time.time()

    try:
        engine = service.engine
        if not engine or not engine.model:
            return BatchHeatmapItem(
                filename=filename,
                success=False,
                error="Model not loaded",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Initialize GradCAM with the model
        gradcam = GradCAM(engine.model)

        # Preprocess image for model
        input_tensor = engine.preprocess(image)

        # Generate heatmap
        heatmap = gradcam(input_tensor)

        # Generate heatmap overlay as base64
        heatmap_base64 = heatmap_to_base64(
            heatmap=heatmap,
            original_image=image,
            colormap=colormap,
            alpha=alpha,
        )

        # Convert original image to base64
        original_base64 = service.processor.to_base64(image)

        # Clean up GradCAM hooks
        gradcam.remove_hooks()

        processing_time_ms = (time.time() - start_time) * 1000

        return BatchHeatmapItem(
            filename=filename,
            success=True,
            heatmap_base64=heatmap_base64,
            original_image_base64=original_base64,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Heatmap generation failed for {filename}: {e}", exc_info=True)
        return BatchHeatmapItem(
            filename=filename,
            success=False,
            error=str(e),
            processing_time_ms=(time.time() - start_time) * 1000,
        )


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
) -> HeatmapResponse:
    """Generate GradCAM heatmap visualization for a chest X-ray."""
    # Validate file type and service availability
    service.validator.validate_or_raise(file)
    service.check_ready_or_raise()

    # Read and validate image
    image = await service.processor.read_from_upload(file, convert_rgb=True)
    logger.info(f"Generating heatmap for: {file.filename}, size: {image.size}")

    # Generate heatmap
    result = _generate_single_heatmap(
        image=image,
        filename=file.filename or "unknown",
        service=service,
        colormap=colormap,
        alpha=alpha,
    )

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=f"Heatmap generation failed: {result.error}",
        )

    return HeatmapResponse(
        success=True,
        filename=result.filename,
        heatmap_base64=result.heatmap_base64,
        original_image_base64=result.original_image_base64,
        processing_time_ms=result.processing_time_ms,
    )


@router.post("/heatmap-batch", response_model=BatchHeatmapResponse)
async def generate_heatmaps_batch(
    files: List[UploadFile] = File(
        ..., description="Chest X-ray image files (PNG, JPEG)"
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
) -> BatchHeatmapResponse:
    """Generate GradCAM heatmaps for multiple chest X-rays."""
    batch_start_time = time.time()

    max_batch_size = 500
    if len(files) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {max_batch_size} images per batch for heatmap generation.",
        )

    service.check_ready_or_raise()

    results = []

    for file in files:
        try:
            service.validator.validate_or_raise(file)
            image = await service.processor.read_from_upload(file, convert_rgb=True)

            result = _generate_single_heatmap(
                image=image,
                filename=file.filename or "unknown",
                service=service,
                colormap=colormap,
                alpha=alpha,
            )
            results.append(result)

        except HTTPException as e:
            results.append(
                BatchHeatmapItem(
                    filename=file.filename or "unknown",
                    success=False,
                    error=e.detail,
                    processing_time_ms=0.0,
                )
            )
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append(
                BatchHeatmapItem(
                    filename=file.filename or "unknown",
                    success=False,
                    error=str(e),
                    processing_time_ms=0.0,
                )
            )

    total_processing_time_ms = (time.time() - batch_start_time) * 1000

    return BatchHeatmapResponse(
        success=True,
        results=results,
        total_processing_time_ms=total_processing_time_ms,
    )
