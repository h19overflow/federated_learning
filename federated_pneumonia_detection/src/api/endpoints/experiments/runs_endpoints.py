"""
Endpoints for retrieving training run results from database.

Simple REST API to fetch metrics and results using run_id.
Maps database schema to frontend ExperimentResults format.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any
import json
import csv
from io import StringIO, BytesIO
from datetime import datetime

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from .utils import _transform_run_to_results, _find_best_epoch

router = APIRouter(
    prefix="/api/runs",
    tags=["runs", "results"],
)

logger = get_logger(__name__)


@router.get("/debug/all")
async def list_all_runs() -> Dict[str, Any]:
    """
    Debug endpoint: List all runs in database.

    Returns:
        List of all runs with basic info
    """
    db = get_session()

    try:
        runs = db.query(Run).all()

        return {
            "total_runs": len(runs),
            "runs": [
                {
                    "id": run.id,
                    "experiment_id": run.experiment_id,
                    "status": run.status,
                    "training_mode": run.training_mode,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "metrics_count": len(run.metrics) if hasattr(run, 'metrics') else 0
                }
                for run in runs
            ]
        }
    except Exception as e:
        logger.error(f"Error listing runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


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


