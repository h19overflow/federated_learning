"""Inference endpoint Pydantic schemas.

This module contains data models for X-ray inference API requests and responses,
including clinical interpretation from the AI agent.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class PredictionClass(str, Enum):
    """Prediction classification result."""
    NORMAL = "NORMAL"
    PNEUMONIA = "PNEUMONIA"


class InferencePrediction(BaseModel):
    """Core prediction result from the model.

    Attributes:
        predicted_class: The predicted classification (NORMAL or PNEUMONIA).
        confidence: Model confidence score (0.0 to 1.0).
        pneumonia_probability: Raw probability of pneumonia.
        normal_probability: Raw probability of normal.
    """
    predicted_class: PredictionClass
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    pneumonia_probability: float = Field(ge=0.0, le=1.0)
    normal_probability: float = Field(ge=0.0, le=1.0)


class RiskAssessment(BaseModel):
    """Clinical risk assessment from the interpretation agent.

    Attributes:
        risk_level: Overall risk level (LOW, MODERATE, HIGH, CRITICAL).
        false_negative_risk: Estimated risk of false negative.
        factors: List of factors contributing to the assessment.
    """
    risk_level: str = Field(description="LOW, MODERATE, HIGH, or CRITICAL")
    false_negative_risk: str = Field(description="Risk of missed pneumonia case")
    factors: List[str] = Field(default_factory=list)


class ClinicalInterpretation(BaseModel):
    """AI-generated clinical interpretation of the prediction.

    Attributes:
        summary: Brief clinical summary of the finding.
        confidence_explanation: Explanation of model confidence level.
        risk_assessment: Detailed risk assessment.
        recommendations: Clinical recommendations based on findings.
        disclaimer: Medical disclaimer for AI-generated content.
    """
    summary: str
    confidence_explanation: str
    risk_assessment: RiskAssessment
    recommendations: List[str] = Field(default_factory=list)
    disclaimer: str = Field(
        default="This is an AI-assisted interpretation and should not replace "
        "professional medical diagnosis. Always consult a qualified radiologist."
    )


class InferenceResponse(BaseModel):
    """Complete response from the inference endpoint.

    Attributes:
        success: Whether the inference completed successfully.
        prediction: Core model prediction results.
        clinical_interpretation: Optional AI-generated clinical analysis.
        model_version: Version/checkpoint of the model used.
        processing_time_ms: Time taken for inference in milliseconds.
    """
    success: bool = True
    prediction: InferencePrediction
    clinical_interpretation: Optional[ClinicalInterpretation] = None
    model_version: str = Field(default="pneumonia_model_01_0.988-v2")
    processing_time_ms: float = Field(ge=0.0)


class InferenceError(BaseModel):
    """Error response for failed inference requests.

    Attributes:
        success: Always False for error responses.
        error: Error type identifier.
        detail: Human-readable error description.
    """
    success: bool = False
    error: str
    detail: str


class BatchInferenceRequest(BaseModel):
    """Request model for batch inference (future use).

    Attributes:
        include_clinical_interpretation: Whether to generate AI interpretation.
        priority: Processing priority for the batch.
    """
    include_clinical_interpretation: bool = True
    priority: str = Field(default="normal", description="normal, high, or urgent")


class HealthCheckResponse(BaseModel):
    """Health check response for the inference service.

    Attributes:
        status: Service status (healthy, degraded, unhealthy).
        model_loaded: Whether the model is loaded and ready.
        gpu_available: Whether GPU acceleration is available.
        model_version: Currently loaded model version.
    """
    status: str
    model_loaded: bool
    gpu_available: bool
    model_version: Optional[str] = None
