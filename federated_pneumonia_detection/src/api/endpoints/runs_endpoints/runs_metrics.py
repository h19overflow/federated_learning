"""Metrics retrieval endpoints for training runs."""

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.api.deps import get_db
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.models import (
    RunMetric,
    ServerEvaluation,
)
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from ..schema.runs_schemas import MetricsResponse
from .shared.utils import _transform_run_to_results

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/metrics", response_model=MetricsResponse)
async def get_run_metrics(
    run_id: int,
    db: Session = Depends(get_db),
) -> MetricsResponse:
    """
    Get complete training results for a specific run.

    Fetches all metrics from database and transforms to frontend format.
    Uses persisted final epoch stats if available, falls back to on-the-fly calculation.

    Args:
        run_id: Database run ID (received via WebSocket during training)

    Returns:
        ExperimentResults matching frontend TypeScript interface
    """
    try:
        run = run_crud.get_with_metrics(db, run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Try to fetch persisted final epoch stats
        persisted_stats: Optional[Dict[str, float]] = None

        if run.training_mode == "centralized":
            final_metrics = (
                db.query(RunMetric)
                .filter(
                    RunMetric.run_id == run.id,
                    RunMetric.metric_name.in_(
                        [
                            "final_sensitivity",
                            "final_specificity",
                            "final_precision_cm",
                            "final_accuracy_cm",
                            "final_f1_cm",
                        ]
                    ),
                )
                .all()
            )

            if len(final_metrics) == 5:
                persisted_stats = {
                    m.metric_name.replace("final_", ""): m.metric_value
                    for m in final_metrics
                }
                logger.info(
                    f"[Metrics] Using persisted final stats for centralized run {run_id}"
                )

        elif run.training_mode == "federated":
            last_eval = (
                db.query(ServerEvaluation)
                .filter(ServerEvaluation.run_id == run.id)
                .order_by(desc(ServerEvaluation.round_number))
                .first()
            )

            if last_eval and last_eval.additional_metrics:
                persisted_stats = last_eval.additional_metrics.get("final_epoch_stats")
                if persisted_stats:
                    logger.info(
                        f"[Metrics] Using persisted final stats for federated run {run_id}"
                    )

        # Transform with persisted stats (will fall back to calculation if None)
        results = _transform_run_to_results(run, persisted_stats=persisted_stats)

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch run results: {str(e)}",
        )
