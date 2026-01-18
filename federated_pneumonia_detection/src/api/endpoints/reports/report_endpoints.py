"""PDF Report Generation Endpoints.

Provides endpoints for generating professional PDF reports from
inference results, supporting both single image and batch analysis.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from federated_pneumonia_detection.src.api.endpoints.reports.schemas import (
    BatchReportRequest,
    SingleReportRequest,
)
from federated_pneumonia_detection.src.api.endpoints.reports.utils import (
    decode_base64_image,
    prepare_batch_results_for_report,
    prepare_summary_stats_for_report,
)
from federated_pneumonia_detection.src.control.report_generation.pdf_report import (
    generate_batch_summary_report,
    generate_prediction_report,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("/single", response_class=Response)
async def generate_single_report(request: SingleReportRequest) -> Response:
    """Generate a PDF report for a single prediction.

    Args:
        request: Single report request with prediction data

    Returns:
        PDF file as binary response
    """
    try:
        # Decode images if provided
        original_image = None
        if request.original_image_base64:
            original_image = decode_base64_image(request.original_image_base64)

        # Prepare clinical interpretation dict
        clinical_interpretation = None
        if request.clinical_interpretation:
            clinical_interpretation = {
                "summary": request.clinical_interpretation.summary,
                "confidence_explanation": request.clinical_interpretation.confidence_explanation,
                "risk_assessment": request.clinical_interpretation.risk_assessment,
                "recommendations": request.clinical_interpretation.recommendations,
            }

        # Generate PDF
        pdf_bytes = generate_prediction_report(
            prediction_class=request.prediction.predicted_class,
            confidence=request.prediction.confidence,
            pneumonia_probability=request.prediction.pneumonia_probability,
            normal_probability=request.prediction.normal_probability,
            original_image=original_image,
            heatmap_base64=request.heatmap_base64,
            clinical_interpretation=clinical_interpretation,
            filename=request.filename,
            model_version=request.model_version,
            processing_time_ms=request.processing_time_ms,
        )

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pneumonia_report_{timestamp}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        logger.error(f"Failed to generate single report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate report: {str(e)}"
        )


@router.post("/batch", response_class=Response)
async def generate_batch_report(request: BatchReportRequest) -> Response:
    """Generate a PDF report for batch predictions.

    Args:
        request: Batch report request with results and summary

    Returns:
        PDF file as binary response
    """
    try:
        # Prepare data for report generation
        formatted_results = prepare_batch_results_for_report(
            [r.model_dump() for r in request.results]
        )
        summary_stats = prepare_summary_stats_for_report(request.summary.model_dump())

        # Prepare images and heatmaps if included
        images = None
        heatmaps = None
        if request.include_heatmaps:
            images = {}
            heatmaps = {}
            for r in request.results:
                if r.success and r.original_image_base64:
                    img = decode_base64_image(r.original_image_base64)
                    if img:
                        images[r.filename] = img
                if r.success and r.heatmap_base64:
                    heatmap_img = decode_base64_image(r.heatmap_base64)
                    if heatmap_img:
                        heatmaps[r.filename] = heatmap_img

        # Generate PDF
        pdf_bytes = generate_batch_summary_report(
            results=formatted_results,
            summary_stats=summary_stats,
            model_version=request.model_version,
            images=images,
            heatmaps=heatmaps,
        )

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_analysis_report_{timestamp}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        logger.error(f"Failed to generate batch report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate report: {str(e)}"
        )


@router.post("/batch-with-heatmaps", response_class=Response)
async def generate_batch_report_with_heatmaps(request: BatchReportRequest) -> Response:
    """Generate a PDF report for batch predictions with heatmap appendix.

    This endpoint automatically includes heatmaps in the report appendix
    when provided in the request.

    Args:
        request: Batch report request with results, summary, and optional heatmaps

    Returns:
        PDF file as binary response with heatmap appendix
    """
    # Force include_heatmaps to true for this endpoint
    request.include_heatmaps = True
    return await generate_batch_report(request)
