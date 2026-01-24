"""List endpoints for retrieving all training runs."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.api.deps import get_db
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from federated_pneumonia_detection.src.api.deps import get_analytics
from federated_pneumonia_detection.src.control.analytics.internals.backfill_service import (
    BackfillService,
)
from federated_pneumonia_detection.src.control.analytics.facade import AnalyticsFacade
from federated_pneumonia_detection.src.control.analytics.internals.summary_service import (
    SummaryService,
)
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
    if analytics is None:
        logger.warning("Analytics service not available for list endpoint")
        raise HTTPException(
            status_code=503,
            detail="Analytics service unavailable. Please check server logs.",
        )

    # Log applied filters for debugging
    logger.info(
        f"[RunsList] Fetching runs with filters - status: {status}, "
        f"training_mode: {training_mode}, sort_by: {sort_by}, "
        f"sort_order: {sort_order}, limit: {limit}, offset: {offset}"
    )

    try:
        # Use CRUD method for filtered list with pagination
        try:
            runs, total_count = run_crud.list_with_filters(
                db,
                status=status,
                training_mode=training_mode,
                sort_by=sort_by,
                sort_order=sort_order,
                limit=limit,
                offset=offset,
            )
        except Exception as e:
            logger.error(f"[RunsList] Database query failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve runs from database: {str(e)}",
            )

        # Batch fetch final epoch stats for all runs
        run_ids = [r.id for r in runs]
        final_stats_map = {}

        # Batch fetch for centralized runs
        centralized_ids = [r.id for r in runs if r.training_mode == "centralized"]  # type: ignore[misc]
        if centralized_ids:
            try:
                centralized_stats = run_crud.batch_get_final_metrics(
                    db, centralized_ids
                )  # type: ignore[misc]
                final_stats_map.update(centralized_stats)
            except Exception as e:
                logger.error(
                    f"[RunsList] Centralized metrics fetch failed for {len(centralized_ids)} runs: {e}",
                    exc_info=True,
                )
                # Graceful degradation: continue with empty stats for these runs
                for run_id in centralized_ids:
                    final_stats_map[run_id] = {}

        # Batch fetch for federated runs
        federated_ids = [r.id for r in runs if r.training_mode == "federated"]  # type: ignore[misc]
        if federated_ids:
            try:
                federated_stats = run_crud.batch_get_federated_final_stats(
                    db, federated_ids
                )  # type: ignore[misc]
                final_stats_map.update(federated_stats)
            except Exception as e:
                logger.error(
                    f"[RunsList] Federated metrics fetch failed for {len(federated_ids)} runs: {e}",
                    exc_info=True,
                )
                # Graceful degradation: continue with empty stats for these runs
                for run_id in federated_ids:
                    final_stats_map[run_id] = {}

        # Build summaries using SummaryService
        run_summaries = []
        summary_service = SummaryService(cache=None)  # type: ignore[arg-type]
        for run in runs:
            try:
                summary = summary_service._build_run_summary(run, db)  # type: ignore[attr-defined]
                run_summaries.append(summary)
            except Exception as e:
                logger.error(
                    f"[RunsList] Failed to build summary for run {run.id}: {e}",
                    exc_info=True,
                )
                # Graceful degradation: append minimal summary with error indicator
                run_summaries.append(
                    {
                        "id": run.id,
                        "training_mode": run.training_mode,
                        "status": run.status,
                        "start_time": run.start_time.isoformat()
                        if run.start_time
                        else None,
                        "end_time": run.end_time.isoformat() if run.end_time else None,
                        "best_val_recall": 0.0,
                        "best_val_accuracy": 0.0,
                        "metrics_count": 0,
                        "run_description": None,
                        "federated_info": None,
                        "final_epoch_stats": None,
                        "error": "Summary unavailable",
                    }
                )

        return RunsListResponse(
            runs=[RunSummary(**summary) for summary in run_summaries],
            total=total_count,
        )

    except Exception as e:
        logger.error(f"Error listing runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backfill/{run_id}/server-evaluations", response_model=BackfillResponse)
async def backfill_server_evaluations(
    run_id: int,
    db: Session = Depends(get_db),
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

        # Execute backfill using shared service
        backfill_service = BackfillService(
            run_crud_obj=run_crud, server_evaluation_crud_obj=server_evaluation_crud
        )
        result = backfill_service.backfill_from_json(db, run_id, {})

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
