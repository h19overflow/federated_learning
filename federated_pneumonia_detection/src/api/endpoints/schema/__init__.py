"""API schema definitions for endpoints.

Exports schema definitions for chat and runs endpoints, centralizing
all data models in one location following SRP.
"""

from .chat_schemas import (
    ChatHistoryResponse,
    ChatMessage,
    ChatResponse,
    ChatSessionSchema,
    CreateSessionRequest,
)
from .inference_schemas import (
    BatchInferenceResponse,
    BatchSummaryStats,
    ClinicalInterpretation,
    HealthCheckResponse,
    InferenceError,
    InferencePrediction,
    InferenceResponse,
    PredictionClass,
    RiskAssessment,
    SingleImageResult,
)
from .report_schemas import (
    BatchReportRequest,
    BatchResultItem,
    ClinicalInterpretationData,
    PredictionData,
    SingleReportRequest,
)
from .report_schemas import (
    BatchSummaryStats as ReportBatchSummaryStats,
)
from .runs_schemas import (
    AnalyticsSummaryResponse,
    BackfillResponse,
    FederatedInfo,
    ModeMetrics,
    RunDetail,
    RunsListResponse,
    RunSummary,
)

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "ChatHistoryResponse",
    "ChatSessionSchema",
    "CreateSessionRequest",
    "ModeMetrics",
    "RunDetail",
    "AnalyticsSummaryResponse",
    "FederatedInfo",
    "RunSummary",
    "RunsListResponse",
    "BackfillResponse",
    "PredictionClass",
    "InferencePrediction",
    "RiskAssessment",
    "ClinicalInterpretation",
    "InferenceResponse",
    "InferenceError",
    "SingleImageResult",
    "BatchSummaryStats",
    "BatchInferenceResponse",
    "HealthCheckResponse",
    "PredictionData",
    "ClinicalInterpretationData",
    "SingleReportRequest",
    "BatchResultItem",
    "ReportBatchSummaryStats",
    "BatchReportRequest",
]
