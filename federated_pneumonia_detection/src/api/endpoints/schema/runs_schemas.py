"""Pydantic schemas for runs endpoints.

This module contains data models for runs API requests and responses,
separating schema definitions from business logic to follow SRP.
All schemas include comprehensive validation using Pydantic Field constraints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FinalEpochStats(BaseModel):
    """Final epoch confusion matrix statistics."""

    sensitivity: float = Field(ge=0, le=1, description="TP / (TP + FN) - Recall")
    specificity: float = Field(ge=0, le=1, description="TN / (TN + FP)")
    precision_cm: float = Field(ge=0, le=1, description="TP / (TP + FP)")
    accuracy_cm: float = Field(ge=0, le=1, description="(TP + TN) / Total")
    f1_cm: float = Field(ge=0, le=1, description="2 * (P * R) / (P + R)")


class ModeMetrics(BaseModel):
    """Aggregated metrics for a training mode.

    Attributes:
        count: Number of runs in this mode.
        avg_accuracy: Average accuracy across runs (0-1).
        avg_precision: Average precision across runs (0-1).
        avg_recall: Average recall across runs (0-1).
        avg_f1: Average F1-score across runs (0-1).
        avg_duration_minutes: Average training duration in minutes.
    """

    count: int = Field(ge=0, description="Number of runs")
    avg_accuracy: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Average accuracy",
    )
    avg_precision: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Average precision",
    )
    avg_recall: Optional[float] = Field(None, ge=0, le=1, description="Average recall")
    avg_f1: Optional[float] = Field(None, ge=0, le=1, description="Average F1-score")
    avg_duration_minutes: Optional[float] = Field(
        None,
        ge=0,
        description="Average duration in minutes",
    )


class RunDetail(BaseModel):
    """Details of a single run for analytics.

    Attributes:
        run_id: Unique identifier for the run.
        training_mode: Training mode (centralized/federated).
        best_accuracy: Best accuracy achieved during training.
        best_precision: Best precision achieved during training.
        best_recall: Best recall achieved during training.
        best_f1: Best F1-score achieved during training.
        duration_minutes: Total training duration in minutes.
        start_time: ISO format start timestamp.
        status: Current run status.
    """

    run_id: int = Field(gt=0, description="Unique run identifier")
    training_mode: str = Field(description="Training mode")
    best_accuracy: Optional[float] = Field(None, ge=0, le=1)
    best_precision: Optional[float] = Field(None, ge=0, le=1)
    best_recall: Optional[float] = Field(None, ge=0, le=1)
    best_f1: Optional[float] = Field(None, ge=0, le=1)
    duration_minutes: Optional[float] = Field(None, ge=0)
    start_time: Optional[str] = Field(None, description="ISO format timestamp")
    status: str = Field(description="Run status")


class AnalyticsSummaryResponse(BaseModel):
    """Response for analytics summary endpoint.

    Attributes:
        total_runs: Total number of runs across all modes.
        success_rate: Proportion of filtered runs relative to all runs with same status (0-1).
                     This is NOT a success/completion rate, but a filtering ratio showing
                     what fraction of all status-matching runs passed through additional filters.
        centralized: Aggregated metrics for centralized training.
        federated: Aggregated metrics for federated training.
        top_runs: Top performing runs by accuracy.
    """

    total_runs: int = Field(ge=0, description="Total number of runs")
    success_rate: float = Field(
        ge=0,
        le=1,
        description="Proportion of filtered runs to total status-matching runs (filtered_count / all_status_count)",
    )
    centralized: ModeMetrics = Field(description="Centralized mode metrics")
    federated: ModeMetrics = Field(description="Federated mode metrics")
    top_runs: List[RunDetail] = Field(default=[], description="Top performing runs")


class FederatedInfo(BaseModel):
    """Federated-specific run information.

    Attributes:
        num_rounds: Number of training rounds completed.
        num_clients: Number of participating clients.
        has_server_evaluation: Whether server evaluation was enabled.
        best_accuracy: Best accuracy across all rounds.
        best_recall: Best recall across all rounds.
        latest_round: Most recent round number.
        latest_accuracy: Accuracy from the latest round.
    """

    num_rounds: int = Field(ge=0, description="Number of training rounds")
    num_clients: int = Field(ge=0, description="Number of clients")
    has_server_evaluation: bool = Field(description="Server evaluation enabled")
    best_accuracy: Optional[float] = Field(None, ge=0, le=1)
    best_recall: Optional[float] = Field(None, ge=0, le=1)
    latest_round: Optional[int] = Field(None, ge=0)
    latest_accuracy: Optional[float] = Field(None, ge=0, le=1)


class RunSummary(BaseModel):
    """Summary of a single run.

    Attributes:
        id: Unique run identifier.
        training_mode: Training mode (centralized/federated).
        status: Current run status.
        start_time: ISO format start timestamp.
        end_time: ISO format end timestamp.
        best_val_recall: Best validation recall achieved.
        best_val_accuracy: Best validation accuracy achieved.
        metrics_count: Number of metrics recorded.
        run_description: Optional description of the run.
        federated_info: Federated-specific info if applicable.
    """

    id: int = Field(gt=0, description="Run identifier")
    training_mode: str = Field(description="Training mode")
    status: str = Field(description="Run status")
    start_time: Optional[str] = Field(None, description="ISO format timestamp")
    end_time: Optional[str] = Field(None, description="ISO format timestamp")
    best_val_recall: float = Field(ge=0, le=1, description="Best validation recall")
    best_val_accuracy: float = Field(ge=0, le=1, description="Best validation accuracy")
    metrics_count: int = Field(ge=0, description="Number of metrics")
    run_description: Optional[str] = Field(None, description="Run description")
    federated_info: Optional[FederatedInfo] = Field(
        None,
        description="Federated-specific info",
    )
    final_epoch_stats: Optional[FinalEpochStats] = Field(
        None, description="Pre-computed final epoch statistics"
    )


class RunsListResponse(BaseModel):
    """Response for runs list endpoint.

    Attributes:
        runs: List of run summaries.
        total: Total number of available runs.
    """

    runs: List[RunSummary] = Field(default=[], description="List of runs")
    total: int = Field(ge=0, description="Total number of runs")


class BackfillResponse(BaseModel):
    """Response for backfill operation.

    Attributes:
        run_id: Identifier of the backfilled run.
        success: Whether backfill succeeded.
        message: Human-readable status message.
        rounds_processed: Number of rounds processed.
    """

    run_id: int = Field(gt=0, description="Run identifier")
    success: bool = Field(description="Operation success status")
    message: str = Field(description="Status message")
    rounds_processed: int = Field(ge=0, description="Rounds processed")


class MetricsResponse(BaseModel):
    """Response model for GET /{run_id}/metrics endpoint.

    Attributes:
        experiment_id: Unique experiment identifier.
        status: Current run status.
        final_metrics: Final epoch metrics (accuracy, precision, recall, f1, auc, loss).
        training_history: Per-epoch metrics across training.
        total_epochs: Total number of training epochs.
        metadata: Additional metadata including best metrics and timestamps.
        confusion_matrix: Optional confusion matrix from final epoch.
    """

    experiment_id: str = Field(description="Experiment identifier")
    status: str = Field(description="Run status")
    final_metrics: Dict[str, float] = Field(description="Final epoch metrics")
    training_history: List[Dict[str, Any]] = Field(
        description="Per-epoch training history",
    )
    total_epochs: int = Field(ge=0, description="Total epochs")
    metadata: Dict[str, Any] = Field(description="Additional metadata")
    confusion_matrix: Optional[Dict[str, Any]] = Field(
        None,
        description="Confusion matrix",
    )


class FederatedRoundsResponse(BaseModel):
    """Response model for GET /{run_id}/federated-rounds endpoint.

    Attributes:
        is_federated: Whether this is a federated training run.
        num_rounds: Number of federated rounds completed.
        num_clients: Number of participating clients.
        rounds: List of per-round metrics.
    """

    is_federated: bool = Field(description="Federated training flag")
    num_rounds: int = Field(ge=0, description="Number of rounds")
    num_clients: int = Field(ge=0, description="Number of clients")
    rounds: List[Dict[str, Any]] = Field(default=[], description="Per-round metrics")


class ServerEvaluationResponse(BaseModel):
    """Response model for GET /{run_id}/server-evaluation endpoint.

    Attributes:
        run_id: Run identifier.
        is_federated: Whether this is a federated training run.
        has_server_evaluation: Whether server evaluation data exists.
        evaluations: List of server evaluations per round.
        summary: Summary statistics across all rounds.
    """

    run_id: int = Field(gt=0, description="Run identifier")
    is_federated: bool = Field(description="Federated training flag")
    has_server_evaluation: bool = Field(description="Has evaluation data")
    evaluations: List[Dict[str, Any]] = Field(
        default=[],
        description="Server evaluations",
    )
    summary: Dict[str, Any] = Field(default={}, description="Summary statistics")
