"""Business logic utilities for runs analytics endpoints."""

from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.api.endpoints.schema.runs_schemas import (
    AnalyticsSummaryResponse,
    ModeMetrics,
)
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from .shared.metrics import RunAggregator, get_metric_extractor

logger = get_logger(__name__)


def generate_analytics_summary(
    db: Session,
    runs: List,
    status: str,
    training_mode: Optional[str],
    days: Optional[int],
) -> AnalyticsSummaryResponse:
    """
    Generate analytics summary from filtered runs.

    Args:
        db: Database session
        runs: List of run objects to analyze
        status: Filter status used for success rate calculation
        training_mode: Training mode filter applied
        days: Time filter applied (for logging)

    Returns:
        AnalyticsSummaryResponse with aggregated statistics and top runs
    """
    from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud

    from .shared.services import RunRanker

    if days:
        cutoff_date = datetime.now() - timedelta(days=days)
        runs = [r for r in runs if r.start_time and r.start_time >= cutoff_date]

    if not runs:
        logger.warning(
            f"No runs found with filters: status={status}, mode={training_mode}, days={days}",
        )
        return create_empty_response()

    centralized_runs = [r for r in runs if r.training_mode == "centralized"]
    federated_runs = [r for r in runs if r.training_mode == "federated"]

    centralized_stats = RunAggregator.calculate_statistics(db, centralized_runs)
    federated_stats = RunAggregator.calculate_statistics(db, federated_runs)

    all_run_details = [extract_run_detail(db, r) for r in runs]
    all_run_details = [d for d in all_run_details if d is not None]
    top_runs = RunRanker.get_top_runs(all_run_details, metric="best_accuracy", limit=10)

    total_runs = len(runs)
    all_status_runs = run_crud.get_by_status(db, status)
    filtered_run_ratio = (
        total_runs / len(all_status_runs) if len(all_status_runs) > 0 else 0.0
    )

    logger.info(
        f"Analytics summary generated: {total_runs} runs "
        f"({len(centralized_runs)} centralized, {len(federated_runs)} federated). "
        f"Filtering ratio: {filtered_run_ratio:.4f} ({total_runs}/{len(all_status_runs)} status-matching runs)",
    )

    return AnalyticsSummaryResponse(
        total_runs=total_runs,
        success_rate=round(filtered_run_ratio, 4),
        centralized=ModeMetrics(**centralized_stats),
        federated=ModeMetrics(**federated_stats),
        top_runs=top_runs,
    )


def extract_run_detail(db: Session, run) -> Optional[dict]:
    """
    Extract detailed metrics for a single run.

    Args:
        db: Database session
        run: Run object to extract metrics from

    Returns:
        Dictionary with run details or None if metrics unavailable
    """
    try:
        extractor = get_metric_extractor(run)
        accuracy = extractor.get_best_metric(db, run.id, "accuracy")

        if accuracy is None:
            logger.warning(f"No metrics found for run {run.id}, skipping")
            return None

        duration_minutes = None
        if run.start_time and run.end_time:
            duration_minutes = round(
                (run.end_time - run.start_time).total_seconds() / 60,
                2,
            )

        return {
            "run_id": run.id,
            "training_mode": run.training_mode,
            "best_accuracy": round(accuracy, 4) if accuracy else None,
            "best_precision": round(
                extractor.get_best_metric(db, run.id, "precision") or 0,
                4,
            ),
            "best_recall": round(
                extractor.get_best_metric(db, run.id, "recall") or 0,
                4,
            ),
            "best_f1": round(extractor.get_best_metric(db, run.id, "f1_score") or 0, 4),
            "duration_minutes": duration_minutes,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "status": run.status,
        }
    except Exception as e:
        logger.error(f"Error extracting details for run {run.id}: {e}")
        return None


def create_empty_response() -> AnalyticsSummaryResponse:
    """
    Generate empty analytics response when no runs found.

    Returns:
        AnalyticsSummaryResponse with zeroed/null values
    """
    empty_stats = ModeMetrics(
        count=0,
        avg_accuracy=None,
        avg_precision=None,
        avg_recall=None,
        avg_f1=None,
        avg_duration_minutes=None,
    )
    return AnalyticsSummaryResponse(
        total_runs=0,
        success_rate=0.0,
        centralized=empty_stats,
        federated=empty_stats,
        top_runs=[],
    )
