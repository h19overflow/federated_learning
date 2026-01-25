"""Metrics retrieval endpoints for training runs."""

from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.api.deps import get_db, get_analytics
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.models import (
    RunMetric,
    ServerEvaluation,
)
from federated_pneumonia_detection.src.control.analytics.facade import AnalyticsFacade
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from ..schema.runs_schemas import MetricsResponse

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/metrics", response_model=MetricsResponse)
async def get_run_metrics(
    run_id: int,
    db: Session = Depends(get_db),
    analytics: AnalyticsFacade | None = Depends(get_analytics),
) -> MetricsResponse:
    """
    Get complete training results for a specific run.

    Fetches all metrics from database and transforms to frontend format.
    Uses persisted final epoch stats if available, falls back to on-the-fly calculation.

    Args:
        run_id: Database run ID (received via WebSocket during training)
        db: Database session
        analytics: Optional analytics facade for cached metrics retrieval

    Returns:
        ExperimentResults matching frontend TypeScript interface
    """
    if analytics is None:
        logger.warning("Analytics service not available for metrics endpoint")
        raise HTTPException(
            status_code=503,
            detail="Analytics service unavailable. Please check server logs.",
        )

    try:
        # Use analytics service for metrics retrieval
        logger.info(f"[Metrics] Using analytics service for run {run_id}")
        metrics_data = analytics.metrics.get_run_metrics(db, run_id)
        return MetricsResponse(**metrics_data)

    except ValueError as e:
        logger.warning(f"Run {run_id} not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch run results: {str(e)}",
        )
