"""Batch prediction endpoint for pneumonia detection.

Provides endpoint for running inference on multiple chest X-ray images
with aggregated results and summary statistics.
"""

import logging
import time
from io import BytesIO
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from PIL import Image

from federated_pneumonia_detection.src.api.deps import get_inference_service
from federated_pneumonia_detection.src.api.endpoints.inference.inference_utils import (
    _generate_fallback_interpretation,
)
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchInferenceResponse,
    BatchSummaryStats,
    ClinicalInterpretation,
    InferencePrediction,
    PredictionClass,
    RiskAssessment,
    SingleImageResult,
)
from federated_pneumonia_detection.src.boundary.inference_service import (
    InferenceService,
)
from federated_pneumonia_detection.src.control.dl_model.utils.data.wandb_inference_tracker import (
    get_wandb_tracker,
)

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

    Args:
        files: List of uploaded image files.
        include_clinical_interpretation: Whether to generate clinical analysis per image.
        service: Injected inference service.

    Returns:
        BatchInferenceResponse with individual results and summary statistics.
    """
    batch_start_time = time.time()

    if not service.is_ready():
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
        image_start_time = time.time()

        # Validate file type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            results.append(
                SingleImageResult(
                    filename=file.filename or "unknown",
                    success=False,
                    error=f"Invalid file type: {file.content_type}. Must be PNG or JPEG.",
                    processing_time_ms=(time.time() - image_start_time) * 1000,
                )
            )
            continue

        try:
            contents = await file.read()
            image = Image.open(BytesIO(contents))
            logger.info(f"Batch processing: {file.filename}, size: {image.size}")

            predicted_class, confidence, pneumonia_prob, normal_prob = service.predict(
                image
            )

            prediction = InferencePrediction(
                predicted_class=PredictionClass(predicted_class),
                confidence=confidence,
                pneumonia_probability=pneumonia_prob,
                normal_probability=normal_prob,
            )
            successful_predictions.append(prediction)

            clinical_interpretation = None
            if include_clinical_interpretation:
                agent_response = await service.get_clinical_interpretation(
                    predicted_class=predicted_class,
                    confidence=confidence,
                    pneumonia_probability=pneumonia_prob,
                    normal_probability=normal_prob,
                    image_info={"filename": file.filename, "size": image.size},
                )

                if agent_response:
                    clinical_interpretation = ClinicalInterpretation(
                        summary=agent_response.summary,
                        confidence_explanation=agent_response.confidence_explanation,
                        risk_assessment=RiskAssessment(
                            risk_level=agent_response.risk_level,
                            false_negative_risk=agent_response.false_negative_risk,
                            factors=agent_response.risk_factors,
                        ),
                        recommendations=agent_response.recommendations,
                    )
                    if agent_response.risk_level in ["HIGH", "CRITICAL"]:
                        high_risk_count += 1
                else:
                    clinical_interpretation = _generate_fallback_interpretation(
                        prediction
                    )
                    if clinical_interpretation.risk_assessment.risk_level in [
                        "HIGH",
                        "CRITICAL",
                    ]:
                        high_risk_count += 1

            results.append(
                SingleImageResult(
                    filename=file.filename or "unknown",
                    success=True,
                    prediction=prediction,
                    clinical_interpretation=clinical_interpretation,
                    processing_time_ms=(time.time() - image_start_time) * 1000,
                )
            )

        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            results.append(
                SingleImageResult(
                    filename=file.filename or "unknown",
                    success=False,
                    error=str(e),
                    processing_time_ms=(time.time() - image_start_time) * 1000,
                )
            )

    # Calculate summary statistics
    successful_count = len(successful_predictions)
    failed_count = len(files) - successful_count
    normal_count = sum(
        1 for p in successful_predictions if p.predicted_class == PredictionClass.NORMAL
    )
    pneumonia_count = sum(
        1
        for p in successful_predictions
        if p.predicted_class == PredictionClass.PNEUMONIA
    )
    avg_confidence = (
        sum(p.confidence for p in successful_predictions) / successful_count
        if successful_count > 0
        else 0.0
    )
    total_processing_time = sum(r.processing_time_ms for r in results)
    avg_processing_time = total_processing_time / len(results) if results else 0.0

    summary = BatchSummaryStats(
        total_images=len(files),
        successful=successful_count,
        failed=failed_count,
        normal_count=normal_count,
        pneumonia_count=pneumonia_count,
        avg_confidence=avg_confidence,
        avg_processing_time_ms=avg_processing_time,
        high_risk_count=high_risk_count,
    )

    total_batch_time_ms = (time.time() - batch_start_time) * 1000
    model_version = service.engine.model_version if service.engine else "unknown"

    # Log batch summary to W&B
    tracker = get_wandb_tracker()
    if tracker.is_active:
        tracker.log_batch_prediction(
            total_images=len(files),
            successful=successful_count,
            failed=failed_count,
            normal_count=normal_count,
            pneumonia_count=pneumonia_count,
            avg_confidence=avg_confidence,
            avg_processing_time_ms=avg_processing_time,
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
