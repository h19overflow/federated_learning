"""List endpoints for retrieving all training runs."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.api.deps import get_db, get_analytics
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.control.analytics.facade import AnalyticsFacade
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from ..schema.runs_schemas import BackfillResponse, RunsListResponse, RunSummary

router = APIRouter()
logger = get_logger(__name__)


@router.get("/list", response_model=RunsListResponse)
async def list_all_runs(
    limit: int = Query(
        100,
        ge=1,
        le=1000,
        description="Maximum number of runs to return",
    ),
    offset: int = Query(0, ge=0, description="Number of runs to skip"),
    status: Optional[str] = Query(
        None,
        description="Filter by status (e.g., 'completed', 'running')",
    ),
    training_mode: Optional[str] = Query(
        None,
        description="Filter by training mode (e.g., 'centralized', 'federated')",
    ),
    sort_by: str = Query("start_time", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order: 'asc' or 'desc'"),
    db: Session = Depends(get_db),
    analytics: AnalyticsFacade | None = Depends(get_analytics),
) -> RunsListResponse:
    """
    List training runs with pagination and filtering.

    Args:
        limit: Maximum number of runs to return (1-1000).
        offset: Number of runs to skip for pagination.
        status: Filter by run status.
        training_mode: Filter by training mode.
        sort_by: Field to sort results by.
        sort_order: Sort direction (asc or desc).

     Returns:
         RunsListResponse with filtered, paginated run summaries and total count
    """
    if analytics is None or analytics.summary is None:
        logger.warning("Analytics service not available for list endpoint")
        raise HTTPException(
            status_code=503,
            detail="Analytics service unavailable. Please check server logs.",
        )

    logger.info(
        f"[RunsList] Fetching runs - status: {status}, "
        f"training_mode: {training_mode}, sort_by: {sort_by}, "
        f"sort_order: {sort_order}, limit: {limit}, offset: {offset}"
    )

    try:
        result = analytics.summary.list_runs_with_summaries(
            db=db,
            limit=limit,
            offset=offset,
            status=status,
            training_mode=training_mode,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        return RunsListResponse(
            runs=[RunSummary(**summary) for summary in result["runs"]],
            total=result["total"],
        )

    except Exception as e:
        logger.error(f"[RunsList] Error listing runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backfill/{run_id}/server-evaluations", response_model=BackfillResponse)
async def backfill_server_evaluations(
    run_id: int,
    db: Session = Depends(get_db),
    analytics: AnalyticsFacade | None = Depends(get_analytics),
) -> BackfillResponse:
    """
    Backfill server evaluations from results JSON file.

    Args:
        run_id: Database run ID

    Returns:
        BackfillResponse with operation status
    """
    try:
        # Validate run exists
        run = run_crud.get(db, run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Validate analytics and backfill service are available
        if analytics is None or analytics.backfill is None:
            logger.error("[Backfill] Analytics/BackfillService unavailable")
            raise HTTPException(status_code=503, detail="Backfill service unavailable")

        # Execute backfill using cached service from facade
        result = analytics.backfill.backfill_from_json(db, run_id, {})

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])

        return BackfillResponse(**result)

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"[Backfill] Error for run {run_id}: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
