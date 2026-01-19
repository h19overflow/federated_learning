"""PDF Report Generation Facade for Pneumonia Detection Results.

Provides a clean interface for generating professional clinical reports
with X-ray images, heatmaps, predictions, and AI-generated interpretations.

This module implements the Facade pattern to simplify report generation
while delegating to specialized utility modules for the implementation.
"""

import io
from typing import Optional

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate

from federated_pneumonia_detection.src.control.report_generation.internals.sections import (
    build_batch_report,
    build_single_report,
)
from federated_pneumonia_detection.src.control.report_generation.internals.styles import (
    get_styles,
)


def generate_prediction_report(
    prediction_class: str,
    confidence: float,
    pneumonia_probability: float,
    normal_probability: float,
    original_image: Optional[Image.Image] = None,
    heatmap_base64: Optional[str] = None,
    clinical_interpretation: Optional[dict] = None,
    filename: Optional[str] = None,
    model_version: str = "unknown",
    processing_time_ms: float = 0.0,
) -> bytes:
    """Generate a PDF report for a single prediction.

    Args:
        prediction_class: NORMAL or PNEUMONIA
        confidence: Model confidence (0-1)
        pneumonia_probability: Probability of pneumonia (0-1)
        normal_probability: Probability of normal (0-1)
        original_image: Original X-ray image (PIL Image)
        heatmap_base64: GradCAM heatmap as base64 string
        clinical_interpretation: Dict with summary, risk_assessment, recommendations
        filename: Original filename
        model_version: Model version string
        processing_time_ms: Processing time in milliseconds

    Returns:
        PDF file as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm,
    )

    styles = get_styles()
    story = []

    build_single_report(
        story=story,
        styles=styles,
        prediction_class=prediction_class,
        confidence=confidence,
        pneumonia_probability=pneumonia_probability,
        normal_probability=normal_probability,
        original_image=original_image,
        heatmap_base64=heatmap_base64,
        clinical_interpretation=clinical_interpretation,
        filename=filename,
        model_version=model_version,
        processing_time_ms=processing_time_ms,
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_batch_summary_report(
    results: list,
    summary_stats: dict,
    model_version: str = "unknown",
    images: Optional[list] = None,
    heatmaps: Optional[dict] = None,
) -> bytes:
    """Generate a comprehensive PDF report for batch predictions.

    Args:
        results: List of prediction result dicts
        summary_stats: Aggregate statistics dict
        model_version: Model version string
        images: Optional list of (filename, PIL.Image) tuples for appendix
        heatmaps: Optional dict mapping filename to heatmap base64 strings

    Returns:
        PDF file as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18*mm,
        leftMargin=18*mm,
        topMargin=15*mm,
        bottomMargin=15*mm,
    )

    styles = get_styles()
    story = []

    build_batch_report(
        story=story,
        styles=styles,
        results=results,
        summary_stats=summary_stats,
        model_version=model_version,
        images=images,
        heatmaps=heatmaps,
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
