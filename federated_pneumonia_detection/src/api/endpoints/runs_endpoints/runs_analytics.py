"""
Analytics endpoints for aggregated training run statistics.

Provides endpoints for comparing centralized vs federated training performance
with aggregated metrics and top-performing runs. Delegates business logic to
runs_analytics_utils following SOLID principles.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from ..schema.runs_schemas import AnalyticsSummaryResponse
from .runs_analytics_utils import generate_analytics_summary

router = APIRouter()
logger = get_logger(__name__)


@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
async def get_analytics_summary(
    status: Optional[str] = Query("completed", description="Filter by run status"),
    training_mode: Optional[str] = Query(None, description="Filter by training mode"),
    days: Optional[int] = Query(None, description="Filter by last N days"),
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
        runs = run_crud.get_by_status_and_mode(
            db,
            status=status,
            training_mode=training_mode,
        )

        return generate_analytics_summary(db, runs, status, training_mode, days)

    except Exception as e:
        logger.error(f"Error generating analytics summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
