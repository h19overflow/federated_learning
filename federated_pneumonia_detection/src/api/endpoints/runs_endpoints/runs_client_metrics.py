"""Client metrics retrieval endpoints for federated training runs."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.api.deps import get_analytics, get_db
from federated_pneumonia_detection.src.control.analytics.facade import AnalyticsFacade
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from ..schema.runs_schemas import ClientMetricsResponse

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/client-metrics", response_model=ClientMetricsResponse)
async def get_client_metrics(
    run_id: int,
    db: Session = Depends(get_db),
    analytics: AnalyticsFacade | None = Depends(get_analytics),
) -> ClientMetricsResponse:
    """
    Get per-client training metrics for a federated learning run.

    Provides granular metrics grouped by client for visualization of
    individual client training progress in federated learning scenarios.

    Args:
        run_id: Database run ID
        db: Database session
        analytics: Optional analytics facade for cached metrics retrieval

    Returns:
        ClientMetricsResponse with per-client training data

    Raises:
        404: Run not found
        503: Analytics service unavailable
    """
    if analytics is None or analytics.metrics is None:
        logger.warning("Analytics service not available for client metrics endpoint")
        raise HTTPException(
            status_code=503,
            detail="Analytics service unavailable. Please check server logs.",
        )

    try:
        logger.info(f"[ClientMetrics] Fetching client metrics for run {run_id}")
        metrics_data = analytics.metrics.get_client_metrics(db, run_id)
        return ClientMetricsResponse(**metrics_data)

    except ValueError as e:
        logger.warning(f"Run {run_id} not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching client metrics for run {run_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch client metrics: {str(e)}",
        )
