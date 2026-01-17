"""Batch prediction endpoint for pneumonia detection.

Provides endpoint for running inference on multiple chest X-ray images
with aggregated results and summary statistics.
"""

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from federated_pneumonia_detection.src.api.deps import (
    get_inference_engine,
    get_clinical_agent,
)
from federated_pneumonia_detection.src.api.endpoints.inference.inference_utils import (
    calculate_batch_summary,
    process_single_image,
)
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchInferenceResponse,
    InferencePrediction,
    SingleImageResult,
)
from federated_pneumonia_detection.src.control.model_inferance.inference_engine import InferenceEngine
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.clinical import (
    ClinicalInterpretationAgent,
)
from federated_pneumonia_detection.src.control.dl_model.utils.data.wandb_inference_tracker import (
    get_wandb_tracker,
)

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
    engine: Optional[InferenceEngine] = Depends(get_inference_engine),
    clinical_agent: Optional[ClinicalInterpretationAgent] = Depends(get_clinical_agent),
) -> BatchInferenceResponse:
    """Run pneumonia detection on multiple chest X-ray images.

    Processes images sequentially and returns aggregated results with summary statistics.
    Clinical interpretation is disabled by default for batch processing to improve speed.

    Args:
        files: List of uploaded image files.
        include_clinical_interpretation: Whether to generate clinical analysis per image.
        engine: Injected inference engine.
        clinical_agent: Injected clinical agent.

    Returns:
        BatchInferenceResponse with individual results and summary statistics.
    """
    batch_start_time = time.time()

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference model is not available. Please try again later.",
        )

    if len(files) > 500:
        raise HTTPException(
            status_code=400,
            detail="Maximum 500 images allowed per batch request.",
        )

    results: List[SingleImageResult] = []
    successful_predictions: List[InferencePrediction] = []
    high_risk_count = 0

    for file in files:
        result, prediction, is_high_risk = await process_single_image(
            file=file,
            engine=engine,
            clinical_agent=clinical_agent,
            include_clinical_interpretation=include_clinical_interpretation,
        )
        results.append(result)
        if prediction is not None:
            successful_predictions.append(prediction)
        if is_high_risk:
            high_risk_count += 1

    summary = calculate_batch_summary(
        results=results,
        successful_predictions=successful_predictions,
        total_files=len(files),
        high_risk_count=high_risk_count,
    )

    total_batch_time_ms = (time.time() - batch_start_time) * 1000
    model_version = engine.model_version

    tracker = get_wandb_tracker()
    if tracker.is_active:
        tracker.log_batch_prediction(
            total_images=len(files),
            successful=summary.successful,
            failed=summary.failed,
            normal_count=summary.normal_count,
            pneumonia_count=summary.pneumonia_count,
            avg_confidence=summary.avg_confidence,
            avg_processing_time_ms=summary.avg_processing_time_ms,
            total_processing_time_ms=total_batch_time_ms,
            high_risk_count=high_risk_count,
            clinical_interpretation_used=include_clinical_interpretation,
            model_version=model_version,
        )

    return BatchInferenceResponse(
        success=True,
        results=results,
        summary=summary,
        model_version=model_version,
        total_processing_time_ms=total_batch_time_ms,
    )
