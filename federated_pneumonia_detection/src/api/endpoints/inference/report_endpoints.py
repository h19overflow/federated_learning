"""PDF Report generation endpoints for inference results.

Provides endpoints to generate clinical-grade PDF reports for
single predictions and batch analysis summaries.
"""

import logging
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import StreamingResponse
from PIL import Image

from federated_pneumonia_detection.src.api.deps import get_inference_service
from federated_pneumonia_detection.src.boundary.inference_service import InferenceService
from federated_pneumonia_detection.src.control.report_generation import (
    generate_prediction_report,
    generate_batch_summary_report,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/inference", tags=["inference", "reports"])


@router.post("/report/single")
async def generate_single_report(
    file: UploadFile = File(..., description="Chest X-ray image file (PNG, JPEG)"),
    include_heatmap: bool = Form(default=True, description="Include GradCAM heatmap"),
    include_clinical: bool = Form(default=False, description="Include clinical interpretation"),
    service: InferenceService = Depends(get_inference_service),
):
    """Generate a PDF report for a single X-ray prediction.

    Runs inference on the uploaded image and generates a professional
    clinical report with prediction results, heatmap visualization,
    and optional clinical interpretation.

    Args:
        file: Uploaded X-ray image file
        include_heatmap: Whether to include GradCAM heatmap in report
        include_clinical: Whether to include AI clinical interpretation
        service: Injected inference service

    Returns:
        PDF file as streaming response
    """
    # Validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be PNG or JPEG.",
        )

    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Inference model is not available.",
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # Run inference
        import time
        start_time = time.time()
        predicted_class, confidence, pneumonia_prob, normal_prob = service.predict(image)
        processing_time_ms = (time.time() - start_time) * 1000

        # Generate heatmap if requested
        heatmap_base64 = None
        if include_heatmap:
            heatmap_base64 = service.generate_heatmap(image)

        # Get clinical interpretation if requested
        clinical_interpretation = None
        if include_clinical:
            agent_response = await service.get_clinical_interpretation(
                predicted_class=predicted_class,
                confidence=confidence,
                pneumonia_probability=pneumonia_prob,
                normal_probability=normal_prob,
                image_info={"filename": file.filename, "size": image.size},
            )
            if agent_response:
                clinical_interpretation = {
                    "summary": agent_response.summary,
                    "confidence_explanation": agent_response.confidence_explanation,
                    "risk_assessment": {
                        "risk_level": agent_response.risk_level,
                        "false_negative_risk": agent_response.false_negative_risk,
                        "factors": agent_response.risk_factors,
                    },
                    "recommendations": agent_response.recommendations,
                }

        # Get model version
        model_version = service.engine.model_version if service.engine else "unknown"

        # Generate PDF
        pdf_bytes = generate_prediction_report(
            prediction_class=predicted_class,
            confidence=confidence,
            pneumonia_probability=pneumonia_prob,
            normal_probability=normal_prob,
            original_image=image,
            heatmap_base64=heatmap_base64,
            clinical_interpretation=clinical_interpretation,
            filename=file.filename,
            model_version=model_version,
            processing_time_ms=processing_time_ms,
        )

        # Generate filename
        report_filename = f"xray_report_{file.filename.rsplit('.', 1)[0]}.pdf"

        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{report_filename}"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}",
        )


@router.post("/report/batch-summary")
async def generate_batch_report(
    results_json: str = Form(..., description="JSON string of batch results"),
    summary_json: str = Form(..., description="JSON string of summary statistics"),
    service: InferenceService = Depends(get_inference_service),
):
    """Generate a summary PDF report for batch predictions.

    Takes the results from a batch prediction and generates a
    professional summary report with statistics and result table.

    Args:
        results_json: JSON string containing list of prediction results
        summary_json: JSON string containing summary statistics
        service: Injected inference service

    Returns:
        PDF file as streaming response
    """
    import json

    try:
        results = json.loads(results_json)
        summary_stats = json.loads(summary_json)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON format: {str(e)}",
        )

    try:
        model_version = service.engine.model_version if service.engine else "unknown"

        pdf_bytes = generate_batch_summary_report(
            results=results,
            summary_stats=summary_stats,
            model_version=model_version,
        )

        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="batch_analysis_report.pdf"'
            },
        )

    except Exception as e:
        logger.error(f"Batch report generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}",
        )


@router.post("/report/batch-with-images")
async def generate_batch_report_with_images(
    files: list[UploadFile] = File(..., description="X-ray image files"),
    results_json: str = Form(..., description="JSON string of batch results"),
    summary_json: str = Form(..., description="JSON string of summary statistics"),
    include_heatmaps: bool = Form(default=True, description="Generate heatmaps for appendix"),
    max_appendix_images: int = Form(default=10, description="Max images to include in appendix"),
    service: InferenceService = Depends(get_inference_service),
):
    """Generate a comprehensive PDF report with image appendix.

    Creates a professional batch report including an appendix with
    individual X-ray images and their GradCAM heatmaps.

    Args:
        files: Uploaded X-ray image files
        results_json: JSON string containing list of prediction results
        summary_json: JSON string containing summary statistics
        include_heatmaps: Whether to generate heatmaps for each image
        max_appendix_images: Maximum number of images to include in appendix
        service: Injected inference service

    Returns:
        PDF file as streaming response
    """
    import json

    try:
        results = json.loads(results_json)
        summary_stats = json.loads(summary_json)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON format: {str(e)}",
        )

    try:
        model_version = service.engine.model_version if service.engine else "unknown"

        # Process images
        images = []
        heatmaps = {}

        # Limit appendix images
        max_images = min(max_appendix_images, 20, len(files))

        for i, file in enumerate(files[:max_images]):
            try:
                contents = await file.read()
                image = Image.open(BytesIO(contents))

                # Store image
                images.append((file.filename, image))

                # Generate heatmap if requested
                if include_heatmaps and service.is_ready():
                    heatmap_base64 = service.generate_heatmap(image)
                    if heatmap_base64:
                        heatmaps[file.filename] = heatmap_base64

            except Exception as e:
                logger.warning(f"Failed to process image {file.filename}: {e}")
                continue

        pdf_bytes = generate_batch_summary_report(
            results=results,
            summary_stats=summary_stats,
            model_version=model_version,
            images=images,
            heatmaps=heatmaps,
        )

        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="batch_analysis_report_with_appendix.pdf"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch report with images generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}",
        )
