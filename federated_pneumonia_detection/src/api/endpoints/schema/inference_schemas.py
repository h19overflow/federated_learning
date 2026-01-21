"""Inference endpoint Pydantic schemas.

This module contains data models for X-ray inference API requests and responses,
including clinical interpretation from the AI agent.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


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
        "professional medical diagnosis. Always consult a qualified radiologist.",
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


class SingleImageResult(BaseModel):
    """Result for a single image in batch inference.

    Attributes:
        filename: Original filename of the image.
        success: Whether inference succeeded for this image.
        prediction: Prediction result if successful.
        clinical_interpretation: Optional clinical interpretation.
        error: Error message if inference failed.
        processing_time_ms: Time taken for this image.
    """

    filename: str
    success: bool = True
    prediction: Optional[InferencePrediction] = None
    clinical_interpretation: Optional[ClinicalInterpretation] = None
    error: Optional[str] = None
    processing_time_ms: float = Field(ge=0.0, default=0.0)


class BatchSummaryStats(BaseModel):
    """Aggregate statistics for batch inference results.

    Attributes:
        total_images: Total number of images processed.
        successful: Number of successful predictions.
        failed: Number of failed predictions.
        normal_count: Number of NORMAL predictions.
        pneumonia_count: Number of PNEUMONIA predictions.
        avg_confidence: Average confidence across all predictions.
        avg_processing_time_ms: Average processing time per image.
        high_risk_count: Number of HIGH/CRITICAL risk assessments.
    """

    total_images: int
    successful: int
    failed: int
    normal_count: int
    pneumonia_count: int
    avg_confidence: float = Field(ge=0.0, le=1.0)
    avg_processing_time_ms: float = Field(ge=0.0)
    high_risk_count: int = 0


class BatchInferenceResponse(BaseModel):
    """Complete response from batch inference endpoint.

    Attributes:
        success: Whether batch processing completed.
        results: List of individual image results.
        summary: Aggregate statistics for the batch.
        model_version: Version of the model used.
        total_processing_time_ms: Total time for entire batch.
    """

    success: bool = True
    results: List[SingleImageResult]
    summary: BatchSummaryStats
    model_version: str = Field(default="pneumonia_model_01_0.988-v2")
    total_processing_time_ms: float = Field(ge=0.0)


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


# GradCAM Visualization Schemas


class HeatmapResponse(BaseModel):
    """Response containing GradCAM heatmap visualization.

    Attributes:
        success: Whether heatmap generation succeeded.
        filename: Original filename of the image.
        heatmap_base64: Base64-encoded PNG of heatmap overlay.
        original_image_base64: Base64-encoded original image.
        processing_time_ms: Time taken to generate heatmap.
    """

    success: bool = True
    filename: str
    heatmap_base64: str = Field(description="Base64-encoded PNG of heatmap overlay")
    original_image_base64: str = Field(description="Base64-encoded original image")
    processing_time_ms: float = Field(ge=0.0)


class BatchHeatmapItem(BaseModel):
    """Single heatmap result in batch response.

    Attributes:
        filename: Original filename of the image.
        success: Whether heatmap generation succeeded for this image.
        heatmap_base64: Base64-encoded heatmap if successful.
        original_image_base64: Base64-encoded original image if successful.
        error: Error message if generation failed.
        processing_time_ms: Time taken for this image.
    """

    filename: str
    success: bool = True
    heatmap_base64: Optional[str] = None
    original_image_base64: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: float = Field(ge=0.0, default=0.0)


class BatchHeatmapResponse(BaseModel):
    """Response containing multiple GradCAM heatmap visualizations.

    Attributes:
        success: Whether batch processing completed.
        results: List of individual heatmap results.
        total_processing_time_ms: Total time for entire batch.
    """

    success: bool = True
    results: List[BatchHeatmapItem]
    total_processing_time_ms: float = Field(ge=0.0)
