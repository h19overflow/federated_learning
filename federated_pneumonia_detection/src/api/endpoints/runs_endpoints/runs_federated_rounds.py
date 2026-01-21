"""
Federated rounds metrics endpoint.

Provides endpoint to fetch federated round metrics for visualization.
This endpoint fetches SERVER-SIDE evaluation metrics (centralized test set evaluation).
"""

from fastapi import APIRouter, HTTPException

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from ..schema.runs_schemas import FederatedRoundsResponse

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/federated-rounds", response_model=FederatedRoundsResponse)
async def get_federated_rounds(run_id: int) -> FederatedRoundsResponse:
    """
    Get federated round metrics for visualization.

    Fetches server-side evaluation metrics per round for federated training runs.
    These are centralized evaluations performed by the server on a held-out test set.

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
                    "metrics": {"accuracy": 0.92, "loss": 0.35, "precision": 0.91, ...}
                }
            ]
        }
    """
    db = get_session()

    try:
        run = run_crud.get(db, run_id)

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

        # Fetch server evaluations from the dedicated server_evaluation table
        server_evaluations = server_evaluation_crud.get_by_run(
            db,
            run_id,
            order_by_round=True,
        )

        logger.info(
            f"[FederatedRounds] Run {run_id} - Found {len(server_evaluations)} server evaluations",
        )

        # Get number of clients (count unique client IDs)
        num_clients = 0
        if run.clients:
            num_clients = len(run.clients)

        # Format response - extract metrics from server evaluations
        rounds_list = []
        for eval_record in server_evaluations:
            metrics_dict = {
                "loss": eval_record.loss,
            }

            # Add optional metrics if they exist
            if eval_record.accuracy is not None:
                metrics_dict["accuracy"] = eval_record.accuracy
            if eval_record.precision is not None:
                metrics_dict["precision"] = eval_record.precision
            if eval_record.recall is not None:
                metrics_dict["recall"] = eval_record.recall
            if eval_record.f1_score is not None:
                metrics_dict["f1"] = eval_record.f1_score
            if eval_record.auroc is not None:
                metrics_dict["auroc"] = eval_record.auroc

            rounds_list.append(
                {"round": eval_record.round_number, "metrics": metrics_dict},
            )

        logger.info(
            f"[FederatedRounds] Run {run_id} - Returning {len(rounds_list)} rounds with server evaluation metrics",
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
        logger.error(
            f"Error fetching federated rounds for run {run_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch federated rounds: {str(e)}",
        )
    finally:
        db.close()
