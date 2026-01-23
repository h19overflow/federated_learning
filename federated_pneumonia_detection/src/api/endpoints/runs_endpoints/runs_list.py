"""List endpoints for retrieving all training runs."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from ..schema.runs_schemas import BackfillResponse, RunsListResponse, RunSummary
from .shared.services import BackfillService
from .shared.summary_builder import FederatedRunSummarizer, RunSummaryBuilder

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
    db = get_session()

    try:
        # Use CRUD method for filtered list with pagination
        runs, total_count = run_crud.list_with_filters(
            db,
            status=status,
            training_mode=training_mode,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        )

        # Batch fetch final epoch stats for all runs
        run_ids = [r.id for r in runs]
        final_stats_map = {}

        # Batch fetch for centralized runs
        centralized_ids = [r.id for r in runs if r.training_mode == "centralized"]
        if centralized_ids:
            final_stats_map.update(
                run_crud.batch_get_final_metrics(db, centralized_ids)
            )

        # Batch fetch for federated runs
        federated_ids = [r.id for r in runs if r.training_mode == "federated"]
        if federated_ids:
            final_stats_map.update(
                run_crud.batch_get_federated_final_stats(db, federated_ids)
            )

        # Build summaries with attached stats
        run_summaries = []
        for run in runs:
            summary = RunSummaryBuilder._build_base_summary(run)
            if run.training_mode == "federated":
                summary["federated_info"] = FederatedRunSummarizer.summarize(run, db)
            else:
                summary["federated_info"] = None
            summary["final_epoch_stats"] = final_stats_map.get(run.id)
            run_summaries.append(summary)

        return RunsListResponse(
            runs=[RunSummary(**summary) for summary in run_summaries],
            total=total_count,
        )

    except Exception as e:
        logger.error(f"Error listing runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/backfill/{run_id}/server-evaluations", response_model=BackfillResponse)
async def backfill_server_evaluations(run_id: int) -> BackfillResponse:
    """
    Backfill server evaluations from results JSON file.

    Args:
        run_id: Database run ID

    Returns:
        BackfillResponse with operation status
    """
    db = get_session()

    try:
        # Validate run exists
        run = run_crud.get(db, run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Execute backfill using shared service
        result = BackfillService.backfill_from_json(db, run_id)

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
    finally:
        db.close()
