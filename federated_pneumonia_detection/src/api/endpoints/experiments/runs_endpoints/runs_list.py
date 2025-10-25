"""
List endpoints for retrieving all training runs.

Provides endpoints to list all runs with summary information and debug details.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from federated_pneumonia_detection.src.boundary.engine import get_session, Run
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/list")
async def list_all_runs() -> Dict[str, Any]:
    """
    List all training runs with summary information.

    Returns:
        Dictionary with list of runs including key metrics
    """
    db = get_session()

    try:
        runs = db.query(Run).order_by(Run.start_time.desc()).all()

        run_summaries = []
        for run in runs:
            # Calculate best validation recall from metrics
            best_val_recall = 0.0
            if run.metrics:
                val_recall_metrics = [
                    m.metric_value for m in run.metrics
                    if m.metric_name == 'val_recall'
                ]
                if val_recall_metrics:
                    best_val_recall = max(val_recall_metrics)

            run_summaries.append({
                "id": run.id,
                "training_mode": run.training_mode,
                "status": run.status,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "best_val_recall": best_val_recall,
                "metrics_count": len(run.metrics) if hasattr(run, 'metrics') else 0,
                "run_description": run.run_description,
            })

        return {
            "runs": run_summaries,
            "total": len(run_summaries)
        }

    except Exception as e:
        logger.error(f"Error listing runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/debug/all")
async def debug_list_all_runs() -> Dict[str, Any]:
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
