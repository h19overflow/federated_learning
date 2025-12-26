"""
Debug endpoints for troubleshooting training runs.

Provides detailed debugging information about runs, metrics, and server evaluations.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from federated_pneumonia_detection.src.boundary.engine import get_session, Run
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/debug/epochs")
async def debug_run_epochs(run_id: int) -> Dict[str, Any]:
    """
    Debug endpoint: Show all epochs stored for a run.

    Returns:
        Detailed breakdown of epochs in database
    """
    db = get_session()

    try:
        run = run_crud.get_with_metrics(db, run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Get unique epochs from metrics
        epochs_set = set()
        metrics_by_epoch = {}

        for metric in run.metrics:
            epoch = metric.step
            epochs_set.add(epoch)

            if epoch not in metrics_by_epoch:
                metrics_by_epoch[epoch] = []

            metrics_by_epoch[epoch].append({
                "name": metric.metric_name,
                "value": metric.metric_value,
                "type": metric.dataset_type
            })

        sorted_epochs = sorted(epochs_set)

        return {
            "run_id": run_id,
            "total_unique_epochs": len(sorted_epochs),
            "epoch_range_0indexed": f"{min(sorted_epochs)} to {max(sorted_epochs)}" if sorted_epochs else "N/A",
            "epoch_range_1indexed": f"{min(sorted_epochs)+1} to {max(sorted_epochs)+1}" if sorted_epochs else "N/A",
            "all_epochs_0indexed": sorted_epochs,
            "all_epochs_1indexed": [e+1 for e in sorted_epochs],
            "total_metrics": len(run.metrics),
            "epochs_detail": {
                str(epoch): {
                    "metrics_count": len(metrics_by_epoch[epoch]),
                    "metric_names": [m["name"] for m in metrics_by_epoch[epoch]]
                }
                for epoch in sorted_epochs
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error debugging run {run_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/debug/all")
async def debug_list_all_runs() -> Dict[str, Any]:
    """Debug endpoint: List all runs with basic info."""
    db = get_session()

    try:
        runs = db.query(Run).all()

        return {
            "total_runs": len(runs),
            "runs": [
                {
                    "id": run.id,
                    "status": run.status,
                    "training_mode": run.training_mode,
                    "start_time": run.start_time.isoformat() if run.start_time else None,
                    "metrics_count": len(run.metrics) if hasattr(run, "metrics") else 0,
                    "clients_count": len(run.clients)
                    if hasattr(run, "clients") and run.clients
                    else 0,
                }
                for run in runs
            ],
        }
    except Exception as e:
        logger.error(f"Error in debug list: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/debug/{run_id}/server-evaluations")
async def debug_server_evaluations(run_id: int) -> Dict[str, Any]:
    """Debug endpoint: Check server evaluations for a specific run."""
    db = get_session()

    try:
        from federated_pneumonia_detection.src.boundary.engine import ServerEvaluation

        run = db.query(Run).filter(Run.id == run_id).first()

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        direct_evals = (
            db.query(ServerEvaluation).filter(ServerEvaluation.run_id == run_id).all()
        )
        crud_evals = server_evaluation_crud.get_by_run(db, run_id)

        logger.info(
            f"[DEBUG] Run {run_id}: Direct={len(direct_evals)}, CRUD={len(crud_evals)}"
        )

        return {
            "run_id": run_id,
            "training_mode": run.training_mode,
            "direct_query_count": len(direct_evals),
            "crud_query_count": len(crud_evals),
            "server_evaluations": [
                {
                    "id": e.id,
                    "round_number": e.round_number,
                    "loss": e.loss,
                    "accuracy": e.accuracy,
                    "precision": e.precision,
                    "recall": e.recall,
                    "f1_score": e.f1_score,
                    "auroc": e.auroc,
                    "num_samples": e.num_samples,
                    "evaluation_time": e.evaluation_time.isoformat()
                    if e.evaluation_time
                    else None,
                }
                for e in direct_evals
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error debugging run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
