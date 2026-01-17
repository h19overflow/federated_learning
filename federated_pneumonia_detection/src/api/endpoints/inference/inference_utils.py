"""Utility functions for inference endpoints.

This module provides shared utility functions used across
inference-related endpoints, particularly for clinical interpretation generation.
"""
import logging
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from fastapi import UploadFile
from PIL import Image

from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    BatchSummaryStats,
    ClinicalInterpretation,
    InferencePrediction,
    PredictionClass,
    RiskAssessment,
    SingleImageResult,
)
from federated_pneumonia_detection.src.control.model_inferance.inference_engine import InferenceEngine
from federated_pneumonia_detection.src.control.agentic_systems.multi_agent_systems.clinical import (
    ClinicalInterpretationAgent,
)

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}


def validate_file_type(content_type: Optional[str]) -> bool:
    """Check if the file content type is an allowed image format."""
    return content_type in ALLOWED_CONTENT_TYPES


async def read_image_from_upload(file: UploadFile) -> Image.Image:
    """Read and return a PIL Image from an uploaded file."""
    contents = await file.read()
    return Image.open(BytesIO(contents))


def run_inference(
    engine: InferenceEngine, image: Image.Image
) -> Tuple[InferencePrediction, str, float, float, float]:
    """Run inference and return prediction with raw values.

    Returns:
        Tuple of (prediction, predicted_class, confidence, pneumonia_prob, normal_prob).
    """
    predicted_class, confidence, pneumonia_prob, normal_prob = engine.predict(image)

    prediction = InferencePrediction(
        predicted_class=PredictionClass(predicted_class),
        confidence=confidence,
        pneumonia_probability=pneumonia_prob,
        normal_probability=normal_prob,
    )

    return prediction, predicted_class, confidence, pneumonia_prob, normal_prob


def build_clinical_interpretation_from_agent(agent_response: Any) -> ClinicalInterpretation:
    """Convert agent response to ClinicalInterpretation schema."""
    return ClinicalInterpretation(
        summary=agent_response.summary,
        confidence_explanation=agent_response.confidence_explanation,
        risk_assessment=RiskAssessment(
            risk_level=agent_response.risk_level,
            false_negative_risk=agent_response.false_negative_risk,
            factors=agent_response.risk_factors,
        ),
        recommendations=agent_response.recommendations,
    )


def _generate_fallback_interpretation(prediction: InferencePrediction) -> ClinicalInterpretation:
    """Generate rule-based clinical interpretation as fallback.

    Used when the LLM-based clinical agent is unavailable.
    """
    if prediction.predicted_class == PredictionClass.PNEUMONIA:
        if prediction.confidence >= 0.9:
            risk_level = "HIGH"
            false_negative_risk = "LOW"
        elif prediction.confidence >= 0.7:
            risk_level = "MODERATE"
            false_negative_risk = "LOW"
        else:
            risk_level = "MODERATE"
            false_negative_risk = "MODERATE"
    else:
        if prediction.confidence >= 0.9:
            risk_level = "LOW"
            false_negative_risk = "LOW"
        elif prediction.confidence >= 0.7:
            risk_level = "LOW"
            false_negative_risk = "MODERATE"
        else:
            risk_level = "MODERATE"
            false_negative_risk = "HIGH"

    factors = []
    if prediction.confidence < 0.7:
        factors.append("Low model confidence suggests uncertainty")
    if prediction.predicted_class == PredictionClass.NORMAL and prediction.pneumonia_probability > 0.3:
        factors.append("Elevated pneumonia probability warrants review")
    if prediction.confidence >= 0.9:
        factors.append("High confidence from validated model")

    recommendations = []
    if risk_level in ["HIGH", "CRITICAL"]:
        recommendations.append("Immediate radiologist review recommended")
        recommendations.append("Consider clinical correlation with symptoms")
    elif risk_level == "MODERATE":
        recommendations.append("Radiologist review within 24 hours recommended")
    else:
        recommendations.append("Standard review workflow appropriate")

    if false_negative_risk in ["MODERATE", "HIGH"]:
        recommendations.append("Consider repeat imaging if clinical suspicion persists")

    if prediction.predicted_class == PredictionClass.PNEUMONIA:
        summary = (
            f"Model detects signs consistent with pneumonia with "
            f"{prediction.confidence:.1%} confidence."
        )
    else:
        summary = (
            f"No definitive signs of pneumonia detected. "
            f"Model confidence: {prediction.confidence:.1%}."
        )

    if prediction.confidence >= 0.9:
        confidence_explanation = "High confidence prediction."
    elif prediction.confidence >= 0.7:
        confidence_explanation = "Moderate confidence; radiologist review advised."
    else:
        confidence_explanation = "Lower confidence; expert review recommended."

    return ClinicalInterpretation(
        summary=summary,
        confidence_explanation=confidence_explanation,
        risk_assessment=RiskAssessment(
            risk_level=risk_level,
            false_negative_risk=false_negative_risk,
            factors=factors,
        ),
        recommendations=recommendations,
    )


async def generate_clinical_interpretation(
    clinical_agent: Optional[ClinicalInterpretationAgent],
    prediction: InferencePrediction,
    predicted_class: str,
    confidence: float,
    pneumonia_prob: float,
    normal_prob: float,
    image_info: Dict[str, Any],
) -> Tuple[Optional[ClinicalInterpretation], bool]:
    """Generate clinical interpretation using agent or fallback.

    Returns:
        Tuple of (clinical_interpretation, is_high_risk).
    """
    if clinical_agent is None:
        return None, False

    agent_response = await clinical_agent.interpret(
        predicted_class=predicted_class,
        confidence=confidence,
        pneumonia_probability=pneumonia_prob,
        normal_probability=normal_prob,
        image_info=image_info,
    )

    if agent_response:
        interpretation = build_clinical_interpretation_from_agent(agent_response)
        is_high_risk = agent_response.risk_level in ["HIGH", "CRITICAL"]
    else:
        interpretation = _generate_fallback_interpretation(prediction)
        is_high_risk = interpretation.risk_assessment.risk_level in ["HIGH", "CRITICAL"]

    return interpretation, is_high_risk


async def process_single_image(
    file: UploadFile,
    engine: InferenceEngine,
    clinical_agent: Optional[ClinicalInterpretationAgent],
    include_clinical_interpretation: bool,
) -> Tuple[SingleImageResult, Optional[InferencePrediction], bool]:
    """Process a single image for batch inference.

    Returns:
        Tuple of (result, prediction if successful, is_high_risk).
    """
    image_start_time = time.time()

    if not validate_file_type(file.content_type):
        return (
            SingleImageResult(
                filename=file.filename or "unknown",
                success=False,
                error=f"Invalid file type: {file.content_type}. Must be PNG or JPEG.",
                processing_time_ms=(time.time() - image_start_time) * 1000,
            ),
            None,
            False,
        )

    try:
        image = await read_image_from_upload(file)
        logger.info(f"Processing: {file.filename}, size: {image.size}")

        prediction, predicted_class, confidence, pneumonia_prob, normal_prob = run_inference(
            engine, image
        )

        clinical_interpretation = None
        is_high_risk = False

        if include_clinical_interpretation:
            clinical_interpretation, is_high_risk = await generate_clinical_interpretation(
                clinical_agent,
                prediction,
                predicted_class,
                confidence,
                pneumonia_prob,
                normal_prob,
                {"filename": file.filename, "size": image.size},
            )

        return (
            SingleImageResult(
                filename=file.filename or "unknown",
                success=True,
                prediction=prediction,
                clinical_interpretation=clinical_interpretation,
                processing_time_ms=(time.time() - image_start_time) * 1000,
            ),
            prediction,
            is_high_risk,
        )

    except Exception as e:
        logger.error(f"Failed to process {file.filename}: {e}")
        return (
            SingleImageResult(
                filename=file.filename or "unknown",
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - image_start_time) * 1000,
            ),
            None,
            False,
        )


def calculate_batch_summary(
    results: List[SingleImageResult],
    successful_predictions: List[InferencePrediction],
    total_files: int,
    high_risk_count: int,
) -> BatchSummaryStats:
    """Calculate summary statistics for batch prediction results."""
    successful_count = len(successful_predictions)
    failed_count = total_files - successful_count

    normal_count = sum(
        1 for p in successful_predictions if p.predicted_class == PredictionClass.NORMAL
    )
    pneumonia_count = sum(
        1 for p in successful_predictions if p.predicted_class == PredictionClass.PNEUMONIA
    )

    avg_confidence = (
        sum(p.confidence for p in successful_predictions) / successful_count
        if successful_count > 0
        else 0.0
    )

    total_processing_time = sum(r.processing_time_ms for r in results)
    avg_processing_time = total_processing_time / len(results) if results else 0.0

    return BatchSummaryStats(
        total_images=total_files,
        successful=successful_count,
        failed=failed_count,
        normal_count=normal_count,
        pneumonia_count=pneumonia_count,
        avg_confidence=avg_confidence,
        avg_processing_time_ms=avg_processing_time,
        high_risk_count=high_risk_count,
    )
