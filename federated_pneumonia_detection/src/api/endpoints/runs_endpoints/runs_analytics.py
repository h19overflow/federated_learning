"""
Analytics endpoints for aggregated training run statistics.

Provides endpoints for comparing centralized vs federated training performance
with aggregated metrics and top-performing runs. Delegates business logic to
runs_analytics_utils following SOLID principles.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.api.deps import get_db, get_analytics
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger
from federated_pneumonia_detection.src.control.analytics.facade import AnalyticsFacade

from ..schema.runs_schemas import AnalyticsSummaryResponse

router = APIRouter()
logger = get_logger(__name__)


@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
async def get_analytics_summary(
    status: Optional[str] = Query("completed", description="Filter by run status"),
    training_mode: Optional[str] = Query(None, description="Filter by training mode"),
    days: Optional[int] = Query(None, description="Filter by last N days"),
    db: Session = Depends(get_db),
    analytics: AnalyticsFacade | None = Depends(get_analytics),
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
    try:
        # Check if analytics service is available
        if analytics is None:
            logger.warning("Analytics service not available, returning empty response")
            raise HTTPException(
                status_code=503,
                detail="Analytics service unavailable. Please check server logs.",
            )

        # Use cached analytics service
        filters = {"status": status, "training_mode": training_mode, "days": days}
        return analytics.metrics.get_analytics_summary(db, filters=filters)  # type: ignore[misc]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating analytics summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
