"""Single image prediction endpoint for pneumonia detection.

Provides endpoint for running inference on individual chest X-ray images
with optional AI-generated clinical interpretation.
"""

import logging
import time
from io import BytesIO

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from PIL import Image

from federated_pneumonia_detection.src.api.deps import get_inference_service
from federated_pneumonia_detection.src.api.endpoints.inference.inference_utils import (
    _generate_fallback_interpretation,
)
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    ClinicalInterpretation,
    InferencePrediction,
    InferenceResponse,
    PredictionClass,
    RiskAssessment,
)
from federated_pneumonia_detection.src.boundary.inference_service import (
    InferenceService,
)
from federated_pneumonia_detection.src.control.dl_model.utils.data.wandb_inference_tracker import (
    get_wandb_tracker,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.post("/predict", response_model=InferenceResponse)
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image file (PNG, JPEG)"),
    include_clinical_interpretation: bool = Query(
        default=True,
        description="Whether to include AI-generated clinical interpretation",
    ),
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    """Run pneumonia detection on an uploaded chest X-ray image.

    Accepts PNG or JPEG images. Returns prediction with confidence scores
    and optional clinical interpretation from the AI agent.

    Args:
        file: Uploaded image file.
        include_clinical_interpretation: Whether to generate clinical analysis.
        service: Injected inference service.

    Returns:
        InferenceResponse with prediction and optional interpretation.

    Raises:
        HTTPException: If model is unavailable or image processing fails.
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
        image = Image.open(BytesIO(contents))
        logger.info(
            f"Processing image: {file.filename}, size: {image.size}, mode: {image.mode}"
        )

    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}",
        )

    try:
        # Run inference via service
        predicted_class, confidence, pneumonia_prob, normal_prob = service.predict(
            image
        )

        prediction = InferencePrediction(
            predicted_class=PredictionClass(predicted_class),
            confidence=confidence,
            pneumonia_probability=pneumonia_prob,
            normal_probability=normal_prob,
        )

        # Generate clinical interpretation if requested
        clinical_interpretation = None
        if include_clinical_interpretation:
            agent_response = await service.get_clinical_interpretation(
                predicted_class=predicted_class,
                confidence=confidence,
                pneumonia_probability=pneumonia_prob,
                normal_probability=normal_prob,
                image_info={
                    "filename": file.filename,
                    "size": image.size,
                },
            )

            if agent_response:
                # Convert agent response to schema
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
            else:
                # Fallback to rule-based interpretation
                clinical_interpretation = _generate_fallback_interpretation(prediction)

        processing_time_ms = (time.time() - start_time) * 1000
        model_version = service.engine.model_version if service.engine else "unknown"

        # Log to W&B
        tracker = get_wandb_tracker()
        if tracker.is_active:
            tracker.log_single_prediction(
                predicted_class=predicted_class,
                confidence=confidence,
                pneumonia_probability=pneumonia_prob,
                normal_probability=normal_prob,
                processing_time_ms=processing_time_ms,
                clinical_interpretation_used=clinical_interpretation is not None,
                model_version=model_version,
            )

        return InferenceResponse(
            success=True,
            prediction=prediction,
            clinical_interpretation=clinical_interpretation,
            model_version=model_version,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        # Log error to W&B
        tracker = get_wandb_tracker()
        if tracker.is_active:
            tracker.log_error("inference", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )
