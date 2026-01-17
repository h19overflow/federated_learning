"""Single image prediction endpoint for pneumonia detection.

Provides endpoint for running inference on individual chest X-ray images
with optional AI-generated clinical interpretation.
"""
import logging
import time
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends

from federated_pneumonia_detection.src.api.deps import (
    get_inference_engine,
    get_clinical_agent,
)
from federated_pneumonia_detection.src.api.endpoints.inference.inference_utils import (
    generate_clinical_interpretation,
    read_image_from_upload,
    run_inference,
    validate_file_type,
)
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    InferenceResponse,
)
from federated_pneumonia_detection.src.control.model_inferance.inference_engine import InferenceEngine
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.clinical import (
    ClinicalInterpretationAgent,
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
    include_heatmap: bool = Query(
        default=True,
        description="Whether to include GradCAM heatmap visualization",
    ),
    engine: Optional[InferenceEngine] = Depends(get_inference_engine),
    clinical_agent: Optional[ClinicalInterpretationAgent] = Depends(get_clinical_agent),
) -> InferenceResponse:
    """Run pneumonia detection on an uploaded chest X-ray image.

    Accepts PNG or JPEG images. Returns prediction with confidence scores
    and optional clinical interpretation from the AI agent.
    """
    start_time = time.time()

    if not validate_file_type(file.content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be PNG or JPEG.",
        )

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference model is not available. Please try again later.",
        )

    try:
        image = await read_image_from_upload(file)
        logger.info(f"Processing image: {file.filename}, size: {image.size}, mode: {image.mode}")
    except Exception as e:
        logger.error(f"Failed to read image: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}",
        )

    try:
        prediction, predicted_class, confidence, pneumonia_prob, normal_prob = run_inference(
            engine, image
        )

        clinical_interpretation = None
        if include_clinical_interpretation:
            clinical_interpretation, _ = await generate_clinical_interpretation(
                clinical_agent,
                prediction,
                predicted_class,
                confidence,
                pneumonia_prob,
                normal_prob,
                {"filename": file.filename, "size": image.size},
            )

        heatmap_base64 = None
        if include_heatmap:
            heatmap_base64 = engine.generate_heatmap(image)

        processing_time_ms = (time.time() - start_time) * 1000
        model_version = engine.model_version

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
            heatmap_base64=heatmap_base64,
            model_version=model_version,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        tracker = get_wandb_tracker()
        if tracker.is_active:
            tracker.log_error("inference", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        )


@router.post("/heatmap")
async def generate_heatmap(
    file: UploadFile = File(..., description="Chest X-ray image file (PNG, JPEG)"),
    engine: Optional[InferenceEngine] = Depends(get_inference_engine),
) -> dict:
    """Generate GradCAM heatmap for an uploaded chest X-ray image.

    This endpoint is optimized for on-demand heatmap generation,
    useful for batch mode where heatmaps aren't pre-generated.
    """
    if not validate_file_type(file.content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be PNG or JPEG.",
        )

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference model is not available.",
        )

    try:
        image = await read_image_from_upload(file)
        heatmap_base64 = engine.generate_heatmap(image)

        if heatmap_base64 is None:
            raise HTTPException(
                status_code=500,
                detail="Heatmap generation unavailable.",
            )

        return {"heatmap_base64": heatmap_base64}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Heatmap generation failed: {str(e)}",
        )
