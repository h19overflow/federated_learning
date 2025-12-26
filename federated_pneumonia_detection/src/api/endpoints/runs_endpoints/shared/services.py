"""Shared services for backfill and ranking operations."""

import json
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional

from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

logger = get_logger(__name__)


class BackfillService:
    """Orchestrates backfill of server evaluations from JSON result files."""

    @staticmethod
    def backfill_from_json(db: Any, run_id: int) -> Dict[str, Any]:
        """Backfill server evaluations from results JSON file."""
        try:
            json_file = BackfillService._load_json_file(run_id)
            server_evals = json_file.get("evaluate_metrics_serverapp", {})

            if not server_evals:
                return {
                    "run_id": run_id,
                    "success": False,
                    "message": "No server evaluations found in JSON file",
                    "rounds_processed": 0,
                }

            logger.info(
                f"[Backfill] Found {len(server_evals)} server evaluation rounds"
            )

            rounds_processed = 0
            for round_num_str, metric_str in server_evals.items():
                round_num = int(round_num_str)
                logger.info(f"[Backfill] Processing round {round_num}...")

                metrics_dict = BackfillService._parse_metrics_string(metric_str)
                if metrics_dict is None:
                    continue

                extracted_metrics = BackfillService._extract_metrics(metrics_dict)

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

        except Exception as err:
            logger.error(f"[Backfill] Failed: {err}")
            return {
                "run_id": run_id,
                "success": False,
                "message": str(err),
                "rounds_processed": 0,
            }

    @staticmethod
    def _load_json_file(run_id: int) -> Dict[str, Any]:
        """Load JSON file for a given run ID."""
        json_file = Path(f"results_{run_id}.json")

        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        logger.info(f"[Backfill] Loading {json_file}...")
        with open(json_file, "r") as f:
            return json.load(f)

    @staticmethod
    def _parse_metrics_string(metric_str: str) -> Optional[Dict[str, Any]]:
        """Parse string representation of metrics dict using ast.literal_eval."""
        try:
            return ast.literal_eval(metric_str)
        except Exception as parse_err:
            logger.error(f"[Backfill] Failed to parse metric string: {parse_err}")
            return None

    @staticmethod
    def _extract_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize metric names from raw metrics dictionary."""
        return {
            "loss": metrics_dict.get("server_loss", 0.0),
            "accuracy": metrics_dict.get("server_accuracy"),
            "precision": metrics_dict.get("server_precision"),
            "recall": metrics_dict.get("server_recall"),
            "f1_score": metrics_dict.get("server_f1"),
            "auroc": metrics_dict.get("server_auroc"),
        }


class RunRanker:
    """Ranks runs by specified metrics for top-N selection."""

    @staticmethod
    def get_top_runs(
        runs: List[Dict[str, Any]],
        metric: str = "best_accuracy",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get top-N runs sorted by metric descending.

        Args:
            runs: List of run detail dictionaries
            metric: Metric key to sort by (default: best_accuracy)
            limit: Maximum number of runs to return (default: 10)

        Returns:
            Top N runs sorted by metric descending
        """
        return sorted(
            runs,
            key=lambda x: x.get(metric, 0.0),
            reverse=True,
        )[:limit]
