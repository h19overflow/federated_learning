"""Pydantic schemas for report generation endpoints."""

from typing import Optional

from pydantic import BaseModel, Field


class PredictionData(BaseModel):
    """Single prediction data for report generation."""

    predicted_class: str = Field(..., description="NORMAL or PNEUMONIA")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence 0-1")
    pneumonia_probability: float = Field(..., ge=0, le=1)
    normal_probability: float = Field(..., ge=0, le=1)


class ClinicalInterpretationData(BaseModel):
    """Clinical interpretation data for report."""

    summary: Optional[str] = None
    confidence_explanation: Optional[str] = None
    risk_assessment: Optional[dict] = None
    recommendations: Optional[list[str]] = None


class SingleReportRequest(BaseModel):
    """Request body for single image PDF report."""

    prediction: PredictionData
    filename: Optional[str] = None
    model_version: str = "unknown"
    processing_time_ms: float = 0.0
    clinical_interpretation: Optional[ClinicalInterpretationData] = None
    heatmap_base64: Optional[str] = Field(
        None,
        description="GradCAM heatmap as base64 string",
    )
    original_image_base64: Optional[str] = Field(
        None,
        description="Original X-ray image as base64 string",
    )


class BatchResultItem(BaseModel):
    """Single result item in batch report."""

    filename: str
    success: bool
    prediction: Optional[PredictionData] = None
    error: Optional[str] = None
    heatmap_base64: Optional[str] = Field(
        None,
        description="GradCAM heatmap overlay as base64 string",
    )
    original_image_base64: Optional[str] = Field(
        None,
        description="Original X-ray image as base64 string",
    )


class BatchSummaryStats(BaseModel):
    """Summary statistics for batch report."""

    total_images: int
    successful: int
    failed: int
    pneumonia_count: int
    normal_count: int
    avg_confidence: float = Field(..., ge=0, le=1)
    high_risk_count: int = 0


class BatchReportRequest(BaseModel):
    """Request body for batch PDF report."""

    results: list[BatchResultItem]
    summary: BatchSummaryStats
    model_version: str = "unknown"
    include_heatmaps: bool = Field(
        default=False,
        description="Whether to include heatmaps in report appendix",
    )
