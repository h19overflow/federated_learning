"""Single image prediction endpoint for pneumonia detection.

Provides endpoint for running inference on individual chest X-ray images
with optional AI-generated clinical interpretation.
"""

import logging
import time

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from federated_pneumonia_detection.src.api.deps import get_inference_service
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    InferenceResponse,
)
from federated_pneumonia_detection.src.control.model_inferance import InferenceService

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
    """
    start_time = time.time()

    # Validate file type and service availability
    service.validator.validate_or_raise(file)
    service.check_ready_or_raise()

    # Read and validate image
    image = await service.processor.read_from_upload(file)
    logger.info(
        f"Processing image: {file.filename}, size: {image.size}, mode: {image.mode}",
    )

    try:
        # Run inference
        predicted_class, confidence, pneumonia_prob, normal_prob = service.predict(
            image,
        )
        prediction = service.create_prediction(
            predicted_class,
            confidence,
            pneumonia_prob,
            normal_prob,
        )

        # Generate clinical interpretation if requested
        clinical_interpretation = None
        if include_clinical_interpretation:
            clinical_interpretation = await service.interpreter.generate(
                predicted_class=predicted_class,
                confidence=confidence,
                pneumonia_prob=pneumonia_prob,
                normal_prob=normal_prob,
                prediction=prediction,
                image_info={"filename": file.filename, "size": image.size},
            )

        processing_time_ms = (time.time() - start_time) * 1000
        model_version = service.engine.model_version if service.engine else "unknown"

        # Log to W&B
        service.logger.log_single(
            predicted_class=predicted_class,
            confidence=confidence,
            pneumonia_prob=pneumonia_prob,
            normal_prob=normal_prob,
            processing_time_ms=processing_time_ms,
            clinical_used=clinical_interpretation is not None,
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
        service.logger.log_error("inference", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )
