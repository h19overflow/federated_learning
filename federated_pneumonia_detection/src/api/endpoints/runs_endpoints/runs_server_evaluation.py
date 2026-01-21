"""
Server evaluation metrics endpoint.

Provides endpoint to fetch server-side evaluation metrics for federated training visualization.
"""

from fastapi import APIRouter, HTTPException

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from ..schema.runs_schemas import ServerEvaluationResponse

router = APIRouter()
logger = get_logger(__name__)


@router.get("/{run_id}/server-evaluation", response_model=ServerEvaluationResponse)
async def get_server_evaluation(run_id: int) -> ServerEvaluationResponse:
    """
    Get server-side evaluation metrics for a federated training run.

    Fetches centralized evaluation metrics computed on the server after each round.

    Args:
        run_id: Database run ID

    Returns:
        {
            "run_id": int,
            "is_federated": bool,
            "has_server_evaluation": bool,
            "evaluations": [
                {
                    "round": 1,
                    "loss": 0.35,
                    "accuracy": 0.92,
                    "precision": 0.91,
                    "recall": 0.93,
                    "f1_score": 0.92,
                    "auroc": 0.95,
                    "confusion_matrix": {
                        "true_positives": 450,
                        "true_negatives": 430,
                        "false_positives": 40,
                        "false_negatives": 30
                    },
                    "num_samples": 950,
                    "evaluation_time": "2025-11-04T12:34:56"
                }
            ],
            "summary": {
                "total_rounds": 5,
                "best_accuracy": {"value": 0.95, "round": 3},
                "best_recall": {"value": 0.94, "round": 4},
                "best_f1_score": {"value": 0.93, "round": 3}
            }
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
                "run_id": run_id,
                "is_federated": False,
                "has_server_evaluation": False,
                "evaluations": [],
                "summary": {},
            }

        # Fetch server evaluations
        evaluations = server_evaluation_crud.get_by_run(db, run_id, order_by_round=True)

        if not evaluations:
            logger.info(
                f"[ServerEvaluation] Run {run_id} - No server evaluations found",
            )
            return {
                "run_id": run_id,
                "is_federated": True,
                "has_server_evaluation": False,
                "evaluations": [],
                "summary": {},
            }

        # Format evaluations for response
        evaluations_list = []
        for eval_record in evaluations:
            eval_dict = {
                "round": eval_record.round_number,
                "loss": eval_record.loss,
                "accuracy": eval_record.accuracy,
                "precision": eval_record.precision,
                "recall": eval_record.recall,
                "f1_score": eval_record.f1_score,
                "auroc": eval_record.auroc,
                "num_samples": eval_record.num_samples,
                "evaluation_time": eval_record.evaluation_time.isoformat()
                if eval_record.evaluation_time
                else None,
            }

            # Add confusion matrix if available
            if all(
                [
                    eval_record.true_positives is not None,
                    eval_record.true_negatives is not None,
                    eval_record.false_positives is not None,
                    eval_record.false_negatives is not None,
                ],
            ):
                eval_dict["confusion_matrix"] = {
                    "true_positives": eval_record.true_positives,
                    "true_negatives": eval_record.true_negatives,
                    "false_positives": eval_record.false_positives,
                    "false_negatives": eval_record.false_negatives,
                }

            # Add additional metrics if available
            if eval_record.additional_metrics:
                eval_dict["additional_metrics"] = eval_record.additional_metrics

            evaluations_list.append(eval_dict)

        # Get summary statistics
        summary = server_evaluation_crud.get_summary_stats(db, run_id)

        logger.info(
            f"[ServerEvaluation] Run {run_id} - Returning {len(evaluations_list)} evaluations",
        )

        return {
            "run_id": run_id,
            "is_federated": True,
            "has_server_evaluation": True,
            "evaluations": evaluations_list,
            "summary": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error fetching server evaluation for run {run_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        db.close()
