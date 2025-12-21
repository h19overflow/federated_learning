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

                # Filter out round 0 (initial evaluation) - only count actual training rounds
                # Flower uses round 0 for initial model evaluation before training starts
                training_rounds = [e for e in server_evals if e.round_number > 0]
                num_training_rounds = len(training_rounds)

                logger.info(
                    f"[ListRuns] Run {run.id} (federated): "
                    f"Found {len(server_evals)} server evaluations ({num_training_rounds} training rounds), {num_clients} clients"
                )

                summary["federated_info"] = {
                    "num_rounds": num_training_rounds,
                    "num_clients": num_clients,
                    "has_server_evaluation": len(training_rounds) > 0,
                }

                # Get best metrics from training rounds (excluding round 0)
                if training_rounds:
                    best_accuracy = max(
                        (e.accuracy for e in training_rounds if e.accuracy is not None),
                        default=None,
                    )
                    best_recall = max(
                        (e.recall for e in training_rounds if e.recall is not None),
                        default=None,
                    )
                    latest_eval = training_rounds[-1]

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
                    logger.warning(
                        f"[ListRuns] Run {run.id}: No training round evaluations found!"
                    )
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
                    "status": run.status,
                    "training_mode": run.training_mode,
                    "start_time": run.start_time.isoformat()
                    if run.start_time
                    else None,
                    "metrics_count": len(run.metrics) if hasattr(run, "metrics") else 0,
                    "clients_count": len(run.clients)
                    if hasattr(run, "clients") and run.clients
                    else 0,
                }
                for run in runs
            ],
        }
    except Exception as e:
        logger.error(f"Error listing runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/debug/{run_id}/server-evaluations")
async def debug_server_evaluations(run_id: int) -> Dict[str, Any]:
    """
    Debug endpoint: Check server evaluations for a specific run.

    Returns:
        Detailed information about server evaluations
    """
    db = get_session()

    try:
        from federated_pneumonia_detection.src.boundary.engine import ServerEvaluation

        # Get the run
        run = db.query(Run).filter(Run.id == run_id).first()

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Direct query for server evaluations
        direct_evals = (
            db.query(ServerEvaluation).filter(ServerEvaluation.run_id == run_id).all()
        )

        # Query via CRUD
        crud_evals = server_evaluation_crud.get_by_run(db, run_id)

        logger.info(
            f"[DEBUG] Run {run_id}: Direct query found {len(direct_evals)}, CRUD found {len(crud_evals)}"
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
        logger.error(
            f"Error debugging server evaluations for run {run_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/backfill/{run_id}/server-evaluations")
async def backfill_server_evaluations(run_id: int) -> Dict[str, Any]:
    """
    Backfill server evaluations from results JSON file.

    Args:
        run_id: Database run ID

    Returns:
        Status of backfill operation
    """
    import json
    import ast
    from pathlib import Path

    db = get_session()

    try:
        # Check if run exists
        run = db.query(Run).filter(Run.id == run_id).first()

        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Load JSON file
        json_file = Path(f"results_{run_id}.json")

        if not json_file.exists():
            raise HTTPException(
                status_code=404, detail=f"JSON file not found: {json_file}"
            )

        logger.info(f"[Backfill] Loading {json_file}...")
        with open(json_file, "r") as f:
            data = json.load(f)

        # Get server evaluations
        server_evals = data.get("evaluate_metrics_serverapp", {})

        if not server_evals:
            return {
                "run_id": run_id,
                "success": False,
                "message": "No server evaluations found in JSON file",
                "rounds_processed": 0,
            }

        logger.info(f"[Backfill] Found {len(server_evals)} server evaluation rounds")

        rounds_processed = 0
        for round_num_str, metric_str in server_evals.items():
            round_num = int(round_num_str)
            logger.info(f"[Backfill] Processing round {round_num}...")

            # Parse the string representation
            try:
                metrics_dict = ast.literal_eval(metric_str)
            except Exception as parse_err:
                logger.error(f"[Backfill] Failed to parse metric string: {parse_err}")
                continue

            # Extract metrics
            extracted_metrics = {
                "loss": metrics_dict.get("server_loss", 0.0),
                "accuracy": metrics_dict.get("server_accuracy"),
                "precision": metrics_dict.get("server_precision"),
                "recall": metrics_dict.get("server_recall"),
                "f1_score": metrics_dict.get("server_f1"),
                "auroc": metrics_dict.get("server_auroc"),
            }

            # Create server evaluation record
            server_evaluation_crud.create_evaluation(
                db=db,
                run_id=run_id,
                round_number=round_num,
                metrics=extracted_metrics,
                num_samples=metrics_dict.get("num_samples"),
            )
            logger.info(
                f"[Backfill] [OK] Persisted server evaluation for round {round_num}"
            )
            rounds_processed += 1

        db.commit()
        logger.info(
            f"[Backfill] [OK] SUCCESS: Backfilled {rounds_processed} server evaluations"
        )

        return {
            "run_id": run_id,
            "success": True,
            "message": f"Successfully backfilled {rounds_processed} server evaluations",
            "rounds_processed": rounds_processed,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Backfill] Error: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
