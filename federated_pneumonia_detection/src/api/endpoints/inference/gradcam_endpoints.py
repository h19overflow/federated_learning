"""GradCAM heatmap generation endpoints.

Provides endpoints for generating GradCAM visualizations that highlight
the regions of chest X-rays that contributed most to the model's predictions.
"""

import base64
import logging
import time
from io import BytesIO
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from federated_pneumonia_detection.src.api.deps import get_inference_service
from federated_pneumonia_detection.src.boundary.inference_service import (
    InferenceService,
)
from federated_pneumonia_detection.src.control.model_inferance.gradcam import (
    GradCAM,
    heatmap_to_base64,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["gradcam"])


class HeatmapResponse(BaseModel):
    """Response containing GradCAM heatmap visualization."""

    success: bool = True
    filename: str
    heatmap_base64: str = Field(description="Base64-encoded PNG of heatmap overlay")
    original_image_base64: str = Field(description="Base64-encoded original image")
    processing_time_ms: float = Field(ge=0.0)


class BatchHeatmapItem(BaseModel):
    """Single heatmap result in batch response."""

    filename: str
    success: bool = True
    heatmap_base64: Optional[str] = None
    original_image_base64: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: float = Field(ge=0.0, default=0.0)


class BatchHeatmapResponse(BaseModel):
    """Response containing multiple GradCAM heatmap visualizations."""

    success: bool = True
    results: List[BatchHeatmapItem]
    total_processing_time_ms: float = Field(ge=0.0)


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _generate_single_heatmap(
    image: Image.Image,
    filename: str,
    service: InferenceService,
    colormap: str = "jet",
    alpha: float = 0.4,
) -> BatchHeatmapItem:
    """Generate heatmap for a single image.

    Args:
        image: PIL Image to process
        filename: Original filename
        service: Inference service with loaded model
        colormap: Colormap for heatmap visualization
        alpha: Overlay transparency

    Returns:
        BatchHeatmapItem with heatmap or error
    """
    start_time = time.time()

    try:
        # Get the model from the service
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
        original_base64 = _image_to_base64(image)

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
    """Generate GradCAM heatmap visualization for a chest X-ray.

    The heatmap highlights regions that contributed most to the model's
    prediction, using gradient-weighted class activation mapping.

    Args:
        file: Uploaded X-ray image (PNG or JPEG)
        colormap: Color scheme for heatmap visualization
        alpha: Transparency of heatmap overlay
        service: Injected inference service

    Returns:
        HeatmapResponse with base64-encoded heatmap overlay and original image

    Raises:
        HTTPException: If model is unavailable or image processing fails
    """
    start_time = time.time()

    # Validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be PNG or JPEG.",
        )

    # Check service availability
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Inference model is not available. Please try again later.",
        )

    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        logger.info(f"Generating heatmap for: {file.filename}, size: {image.size}")
    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}",
        )

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
    """Generate GradCAM heatmaps for multiple chest X-rays.

    Processes images sequentially to avoid memory issues.
    Maximum 50 images per batch for heatmap generation.

    Args:
        files: List of uploaded X-ray images (PNG or JPEG)
        colormap: Color scheme for heatmap visualization
        alpha: Transparency of heatmap overlay
        service: Injected inference service

    Returns:
        BatchHeatmapResponse with results for each image

    Raises:
        HTTPException: If model is unavailable or batch is too large
    """
    batch_start_time = time.time()

    # Limit batch size for heatmaps (more memory intensive than predictions)
    max_batch_size = 500
    if len(files) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {max_batch_size} images per batch for heatmap generation.",
        )

    # Check service availability
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Inference model is not available. Please try again later.",
        )

    results = []

    for file in files:
        # Validate file type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            results.append(
                BatchHeatmapItem(
                    filename=file.filename or "unknown",
                    success=False,
                    error=f"Invalid file type: {file.content_type}",
                    processing_time_ms=0.0,
                )
            )
            continue

        try:
            # Read image
            contents = await file.read()
            image = Image.open(BytesIO(contents)).convert("RGB")

            # Generate heatmap
            result = _generate_single_heatmap(
                image=image,
                filename=file.filename or "unknown",
                service=service,
                colormap=colormap,
                alpha=alpha,
            )
            results.append(result)

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
