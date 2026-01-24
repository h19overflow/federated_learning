"""
Service for consolidating final epoch statistics extraction and persistence.

Eliminates duplicate CM extraction logic across:
- Centralized training (db_operations.py)
- Federated strategy (custom_strategy.py)
"""

from typing import Any, Dict, Optional

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from federated_pneumonia_detection.src.boundary.models import (
    RunMetric,
    ServerEvaluation,
)
from ..utils import (
    calculate_summary_statistics,
)


class FinalEpochStatsService:
    """Service for extracting and persisting final epoch statistics."""

    @staticmethod
    def get_cm_centralized(db: Session, run_id: int) -> Optional[Dict[str, Any]]:
        """
        Extract confusion matrix from final epoch of centralized run.

        Queries RunMetric table for the highest epoch with CM data (val_cm_tp, val_cm_tn, etc.)
        and returns consolidated confusion matrix dictionary.

        Args:
            db: Database session
            run_id: Run ID to extract CM from

        Returns:
            Dict with keys: true_positives, true_negatives, false_positives, false_negatives, epoch
            or None if incomplete data
        """
        # Find max epoch that has CM data
        max_epoch = (
            db.query(func.max(RunMetric.step))
            .filter(RunMetric.run_id == run_id, RunMetric.metric_name == "val_cm_tp")
            .scalar()
        )

        if max_epoch is None:
            return None

        # Query all CM metrics for final epoch
        cm_metrics = (
            db.query(RunMetric)
            .filter(
                RunMetric.run_id == run_id,
                RunMetric.step == max_epoch,
                RunMetric.metric_name.in_(
                    ["val_cm_tp", "val_cm_tn", "val_cm_fp", "val_cm_fn"]
                ),
            )
            .all()
        )

        if len(cm_metrics) != 4:
            return None

        # Build dict with standard keys
        cm_dict = {"epoch": max_epoch}
        for m in cm_metrics:
            key_map = {
                "val_cm_tp": "true_positives",
                "val_cm_tn": "true_negatives",
                "val_cm_fp": "false_positives",
                "val_cm_fn": "false_negatives",
            }
            cm_dict[key_map[m.metric_name]] = int(m.metric_value)

        return cm_dict

    @staticmethod
    def get_cm_federated(db: Session, run_id: int) -> Optional[Dict[str, Any]]:
        """
        Extract confusion matrix from final round of federated run.

        Queries ServerEvaluation table for the latest round and extracts CM values
        from the confusion matrix fields.

        Args:
            db: Database session
            run_id: Run ID to extract CM from

        Returns:
            Dict with keys: true_positives, true_negatives, false_positives, false_negatives
            or None if incomplete data
        """
        # Get last ServerEvaluation record
        last_eval = (
            db.query(ServerEvaluation)
            .filter(ServerEvaluation.run_id == run_id)
            .order_by(desc(ServerEvaluation.round_number))
            .first()
        )

        if not last_eval:
            return None

        # Check if all CM values are present
        if all(
            getattr(last_eval, attr) is not None
            for attr in [
                "true_positives",
                "true_negatives",
                "false_positives",
                "false_negatives",
            ]
        ):
            return {
                "true_positives": last_eval.true_positives,
                "true_negatives": last_eval.true_negatives,
                "false_positives": last_eval.false_positives,
                "false_negatives": last_eval.false_negatives,
            }
        return None

    @staticmethod
    def calculate_and_persist_centralized(
        db: Session, run_id: int
    ) -> Optional[Dict[str, float]]:
        """
        Calculate final epoch stats for centralized run and persist to database.

        Extracts CM from RunMetric, calculates summary statistics (sensitivity, specificity, etc.),
        and persists as final_* metrics using run_metric_crud.

        Args:
            db: Database session
            run_id: Run ID to process

        Returns:
            Dict with calculated stats (sensitivity, specificity, precision_cm, accuracy_cm, f1_cm)
            or None if CM extraction failed
        """
        # Extract CM from centralized run
        final_cm = FinalEpochStatsService.get_cm_centralized(db, run_id)
        if not final_cm:
            return None

        # Calculate summary statistics
        stats = calculate_summary_statistics(final_cm)

        # Persist to database
        run_metric_crud.create_final_epoch_stats(db, run_id, stats, final_cm["epoch"])

        return stats

    @staticmethod
    def calculate_and_persist_federated(
        db: Session, run_id: int, round_metrics: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """
        Calculate final epoch stats for federated run and persist to database.

        Extracts CM from aggregated round metrics, calculates summary statistics,
        and persists to ServerEvaluation additional_metrics using server_evaluation_crud.

        Args:
            db: Database session
            run_id: Run ID to process
            round_metrics: Aggregated metrics from final round (must contain cm_tp, cm_tn, cm_fp, cm_fn)

        Returns:
            Dict with calculated stats (sensitivity, specificity, precision_cm, accuracy_cm, f1_cm)
            or None if CM values are missing
        """
        # Check if all CM values present in round metrics
        cm_keys = ["cm_tp", "cm_tn", "cm_fp", "cm_fn"]
        if not all(k in round_metrics for k in cm_keys):
            return None

        # Build CM dict from round metrics
        cm_dict = {
            "true_positives": int(round_metrics["cm_tp"]),
            "true_negatives": int(round_metrics["cm_tn"]),
            "false_positives": int(round_metrics["cm_fp"]),
            "false_negatives": int(round_metrics["cm_fn"]),
        }

        # Calculate summary statistics
        stats = calculate_summary_statistics(cm_dict)

        # Persist to database
        server_evaluation_crud.update_final_epoch_stats(db, run_id, stats)

        return stats
