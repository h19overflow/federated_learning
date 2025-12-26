"""
List endpoints for retrieving all training runs.

Provides endpoints to list all runs with summary information and backfill operations.
Uses shared modules for summary building and backfill services following SOLID principles.
"""

from fastapi import APIRouter, HTTPException

from federated_pneumonia_detection.src.boundary.engine import get_session, Run
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

from ..schema.runs_schemas import RunsListResponse, RunSummary, BackfillResponse
from .shared.summary_builder import RunSummaryBuilder
from .shared.services import BackfillService

router = APIRouter()
logger = get_logger(__name__)


@router.get("/list", response_model=RunsListResponse)
async def list_all_runs() -> RunsListResponse:
    """
    List all training runs with summary information.

    Returns:
        RunsListResponse with list of run summaries and total count
    """
    db = get_session()

    try:
        runs = db.query(Run).order_by(Run.start_time.desc()).all()
        run_summaries = [RunSummaryBuilder.build(run, db) for run in runs]

        return RunsListResponse(
            runs=[RunSummary(**summary) for summary in run_summaries],
            total=len(run_summaries)
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
