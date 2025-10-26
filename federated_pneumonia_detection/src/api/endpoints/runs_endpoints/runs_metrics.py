"""
Metrics retrieval endpoints for training runs.

Provides endpoint to fetch complete training results and metrics
for a specific run from the database.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from .utils import _transform_run_to_results

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/metrics")
async def get_run_metrics(run_id: int) -> Dict[str, Any]:
    """
    Get complete training results for a specific run.

    Fetches all metrics from database and transforms to frontend format.

    Args:
        run_id: Database run ID (received via WebSocket during training)

    Returns:
        ExperimentResults matching frontend TypeScript interface
    """
    db = get_session()

    try:
        run = run_crud.get_with_metrics(db, run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Transform database data to frontend format
        results = _transform_run_to_results(run)

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch run results: {str(e)}")
    finally:
        db.close()
