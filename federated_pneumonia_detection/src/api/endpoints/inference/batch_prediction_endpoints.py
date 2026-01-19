"""Batch prediction endpoint for pneumonia detection.

Provides endpoint for running inference on multiple chest X-ray images
with aggregated results and summary statistics.
"""

import logging
import time
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from federated_pneumonia_detection.src.api.deps import get_inference_service
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchInferenceResponse,
)
from federated_pneumonia_detection.src.control.model_inferance import InferenceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.post("/predict-batch", response_model=BatchInferenceResponse)
async def predict_batch(
    files: List[UploadFile] = File(
        ..., description="Multiple chest X-ray images (PNG, JPEG)"
    ),
    include_clinical_interpretation: bool = Query(
        default=False,
        description="Whether to include AI-generated clinical interpretation (slower)",
    ),
    service: InferenceService = Depends(get_inference_service),
) -> BatchInferenceResponse:
    """Run pneumonia detection on multiple chest X-ray images.

    Processes images sequentially and returns aggregated results with summary statistics.
    Clinical interpretation is disabled by default for batch processing to improve speed.
    """
    batch_start_time = time.time()

    service.check_ready_or_raise()

    if len(files) > 500:
        raise HTTPException(
            status_code=400,
            detail="Maximum 500 images allowed per batch request.",
        )

    results = []
    for file in files:
        result = await service.process_single(
            file=file,
            include_clinical=include_clinical_interpretation,
        )
        results.append(result)
        logger.info(f"Batch processing: {file.filename}, success: {result.success}")

    # Calculate summary statistics
    summary = service.batch_stats.calculate(results=results, total_images=len(files))

    total_batch_time_ms = (time.time() - batch_start_time) * 1000
    model_version = service.engine.model_version if service.engine else "unknown"

    # Log batch summary to W&B
    service.logger.log_batch(
        summary=summary,
        total_time_ms=total_batch_time_ms,
        clinical_used=include_clinical_interpretation,
        model_version=model_version,
    )

    return BatchInferenceResponse(
        success=True,
        results=results,
        summary=summary,
        model_version=model_version,
        total_processing_time_ms=total_batch_time_ms,
    )
