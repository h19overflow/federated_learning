"""
Federated rounds metrics endpoint.

Provides endpoint to fetch federated round metrics for visualization.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/federated-rounds")
async def get_federated_rounds(run_id: int) -> Dict[str, Any]:
    """
    Get federated round metrics for visualization.

    Fetches global aggregated metrics per round for federated training runs.

    Args:
        run_id: Database run ID

    Returns:
        {
            "is_federated": bool,
            "num_rounds": int,
            "num_clients": int,
            "rounds": [
                {
                    "round": 1,
                    "metrics": {"accuracy": 0.92, "loss": 0.35, ...}
                }
            ]
        }
    """
    db = get_session()

    try:
        run = run_crud.get_with_metrics(db, run_id)

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Check if this is a federated run
        is_federated = run.training_mode == "federated"

        if not is_federated:
            return {
                "is_federated": False,
                "num_rounds": 0,
                "num_clients": 0,
                "rounds": [],
            }

        # Extract global metrics (those prefixed with 'global_')
        federated_rounds = {}
        num_clients = 0

        # Debug: Log all metrics for this run
        logger.info(
            f"[FederatedRounds] Run {run_id} - Total metrics: {len(run.metrics)}"
        )
        logger.info(
            f"[FederatedRounds] Run {run_id} - Training mode: {run.training_mode}"
        )

        global_metrics_count = 0
        for metric in run.metrics:
            if metric.metric_name.startswith("global_"):
                global_metrics_count += 1
                round_num = metric.step  # 'step' field is used as round number
                metric_base_name = metric.metric_name.replace("global_", "")

                if round_num not in federated_rounds:
                    federated_rounds[round_num] = {}

                federated_rounds[round_num][metric_base_name] = metric.metric_value

        logger.info(
            f"[FederatedRounds] Run {run_id} - Found {global_metrics_count} global metrics"
        )
        logger.info(
            f"[FederatedRounds] Run {run_id} - Extracted {len(federated_rounds)} rounds"
        )

        # Get number of clients (count unique client IDs)
        if run.clients:
            num_clients = len(run.clients)

        # Format response
        rounds_list = []
        for round_num in sorted(federated_rounds.keys()):
            rounds_list.append(
                {"round": round_num, "metrics": federated_rounds[round_num]}
            )

        return {
            "is_federated": True,
            "num_rounds": len(rounds_list),
            "num_clients": num_clients,
            "rounds": rounds_list,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching federated rounds for run {run_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch federated rounds: {str(e)}"
        )
    finally:
        db.close()
