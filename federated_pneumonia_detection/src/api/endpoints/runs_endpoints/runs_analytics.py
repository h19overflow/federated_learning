"""
Analytics endpoints for aggregated training run statistics.

Provides endpoints for comparing centralized vs federated training performance
with aggregated metrics and top-performing runs. Uses shared modules for
metric extraction, aggregation, and ranking following SOLID principles.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime, timedelta

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

from ..schema.runs_schemas import AnalyticsSummaryResponse, RunDetail, ModeMetrics
from .shared.metrics import RunAggregator, get_metric_extractor
from .shared.services import RunRanker

router = APIRouter()
logger = get_logger(__name__)


@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
async def get_analytics_summary(
    status: Optional[str] = Query("completed", description="Filter by run status"),
    training_mode: Optional[str] = Query(None, description="Filter by training mode"),
    days: Optional[int] = Query(None, description="Filter by last N days")
) -> AnalyticsSummaryResponse:
    """
    Get aggregated statistics for centralized vs federated training comparison.

    Args:
        status: Filter by run status (default: "completed")
        training_mode: Filter by mode ("centralized", "federated", or None for all)
        days: Filter by last N days (None for all time)

    Returns:
        AnalyticsSummaryResponse with aggregated statistics and top runs
    """
    db = get_session()

    try:
        # Fetch runs with database-level filtering
        runs = run_crud.get_by_status_and_mode(db, status=status, training_mode=training_mode)

        # Apply time filter if specified
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            runs = [r for r in runs if r.start_time and r.start_time >= cutoff_date]

        if not runs:
            logger.warning(
                f"No runs found with filters: status={status}, mode={training_mode}, days={days}"
            )
            return _create_empty_response()

        # Split runs by training mode
        centralized_runs = [r for r in runs if r.training_mode == "centralized"]
        federated_runs = [r for r in runs if r.training_mode == "federated"]

        # Calculate aggregated statistics using shared aggregator
        centralized_stats = RunAggregator.calculate_statistics(db, centralized_runs)
        federated_stats = RunAggregator.calculate_statistics(db, federated_runs)

        # Extract run details and rank
        all_run_details = [_extract_run_detail(db, r) for r in runs]
        all_run_details = [d for d in all_run_details if d is not None]  # Filter None
        top_runs = RunRanker.get_top_runs(all_run_details, metric="best_accuracy", limit=10)

        # Calculate success rate
        total_runs = len(runs)
        all_status_runs = run_crud.get_by_status(db, status)
        success_rate = total_runs / len(all_status_runs) if len(all_status_runs) > 0 else 0.0

        logger.info(
            f"Analytics summary generated: {total_runs} runs "
            f"({len(centralized_runs)} centralized, {len(federated_runs)} federated)"
        )

        return AnalyticsSummaryResponse(
            total_runs=total_runs,
            success_rate=round(success_rate, 4),
            centralized=ModeMetrics(**centralized_stats),
            federated=ModeMetrics(**federated_stats),
            top_runs=[RunDetail(**run) for run in top_runs]
        )

    except Exception as e:
        logger.error(f"Error generating analytics summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


def _extract_run_detail(db, run) -> Optional[dict]:
    """Extract detailed metrics for a single run using shared extractors."""
    try:
        extractor = get_metric_extractor(run)
        accuracy = extractor.get_best_metric(db, run.id, "accuracy")

        if accuracy is None:
            logger.warning(f"No metrics found for run {run.id}, skipping")
            return None

        duration_minutes = None
        if run.start_time and run.end_time:
            duration_minutes = round((run.end_time - run.start_time).total_seconds() / 60, 2)

        return {
            "run_id": run.id,
            "training_mode": run.training_mode,
            "best_accuracy": round(accuracy, 4) if accuracy else None,
            "best_precision": round(extractor.get_best_metric(db, run.id, "precision") or 0, 4),
            "best_recall": round(extractor.get_best_metric(db, run.id, "recall") or 0, 4),
            "best_f1": round(extractor.get_best_metric(db, run.id, "f1_score") or 0, 4),
            "duration_minutes": duration_minutes,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "status": run.status
        }
    except Exception as e:
        logger.error(f"Error extracting details for run {run.id}: {e}")
        return None


def _create_empty_response() -> AnalyticsSummaryResponse:
    """Generate empty response when no runs found."""
    empty_stats = ModeMetrics(
        count=0,
        avg_accuracy=None,
        avg_precision=None,
        avg_recall=None,
        avg_f1=None,
        avg_duration_minutes=None
    )
    return AnalyticsSummaryResponse(
        total_runs=0,
        success_rate=0.0,
        centralized=empty_stats,
        federated=empty_stats,
        top_runs=[]
    )
