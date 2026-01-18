"""Utility functions for inference endpoints.

This module provides shared utility functions used across
inference-related endpoints, particularly for clinical interpretation generation.
"""
from federated_pneumonia_detection.src.api.endpoints.schema.inference_schemas import (
    PredictionClass,
    InferencePrediction,
    ClinicalInterpretation,
    RiskAssessment,
)


def _generate_fallback_interpretation(prediction: InferencePrediction) -> ClinicalInterpretation:
    """Generate rule-based clinical interpretation as fallback.

    Used when the LLM-based clinical agent is unavailable.
    """
    # Determine risk level based on prediction
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

    # Build factors
    factors = []
    if prediction.confidence < 0.7:
        factors.append("Low model confidence suggests uncertainty")
    if prediction.predicted_class == PredictionClass.NORMAL and prediction.pneumonia_probability > 0.3:
        factors.append("Elevated pneumonia probability warrants review")
    if prediction.confidence >= 0.9:
        factors.append("High confidence from validated model")

    # Build recommendations
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

    # Build summary
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

    # Confidence explanation
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
