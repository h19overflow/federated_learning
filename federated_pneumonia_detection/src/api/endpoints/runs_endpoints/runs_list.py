"""
List endpoints for retrieving all training runs.

Provides endpoints to list all runs with summary information and debug details.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from federated_pneumonia_detection.src.boundary.engine import get_session, Run
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
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
            is_federated = run.training_mode == "federated"

            # Calculate best validation recall from metrics
            best_val_recall = 0.0
            if run.metrics:
                val_recall_metrics = [
                    m.metric_value for m in run.metrics if m.metric_name == "val_recall"
                ]
                if val_recall_metrics:
                    best_val_recall = max(val_recall_metrics)

            # Prepare base summary
            summary = {
                "id": run.id,
                "training_mode": run.training_mode,
                "status": run.status,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "best_val_recall": best_val_recall,
                "metrics_count": len(run.metrics) if hasattr(run, "metrics") else 0,
                "run_description": run.run_description,
            }

            # Add federated-specific information
            if is_federated:
                # Get server evaluation summary
                server_evals = server_evaluation_crud.get_by_run(db, run.id)
                num_clients = len(run.clients) if run.clients else 0

                logger.info(
                    f"[ListRuns] Run {run.id} (federated): "
                    f"Found {len(server_evals)} server evaluations, {num_clients} clients"
                )

                summary["federated_info"] = {
                    "num_rounds": len(server_evals),
                    "num_clients": num_clients,
                    "has_server_evaluation": len(server_evals) > 0,
                }

                # Get best metrics from server evaluations
                if server_evals:
                    best_accuracy = max(
                        (e.accuracy for e in server_evals if e.accuracy is not None), 
                        default=None
                    )
                    best_recall = max(
                        (e.recall for e in server_evals if e.recall is not None), 
                        default=None
                    )
                    latest_eval = server_evals[-1]

                    summary["federated_info"]["best_accuracy"] = best_accuracy
                    summary["federated_info"]["best_recall"] = best_recall
                    summary["federated_info"]["latest_round"] = latest_eval.round_number
                    summary["federated_info"]["latest_accuracy"] = latest_eval.accuracy

                    logger.info(
                        f"[ListRuns] Run {run.id} metrics: "
                        f"best_accuracy={best_accuracy}, "
                        f"latest_accuracy={latest_eval.accuracy}, "
                        f"latest_round={latest_eval.round_number}"
                    )
                else:
                    logger.warning(f"[ListRuns] Run {run.id}: No server evaluations found!")
            else:
                summary["federated_info"] = None

            run_summaries.append(summary)

        return {"runs": run_summaries, "total": len(run_summaries)}

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
                    "start_time": run.start_time.isoformat()
                    if run.start_time
                    else None,
                    "metrics_count": len(run.metrics) if hasattr(run, "metrics") else 0,
                }
                for run in runs
            ],
        }
    except Exception as e:
        logger.error(f"Error listing runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
