"""
Debug endpoints for troubleshooting training runs.

Provides detailed debugging information about runs and their metrics,
useful for inspecting epoch data and run structure.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
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
