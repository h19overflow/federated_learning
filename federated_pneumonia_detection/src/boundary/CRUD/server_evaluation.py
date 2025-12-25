from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
from federated_pneumonia_detection.src.boundary.engine import ServerEvaluation
from federated_pneumonia_detection.src.boundary.CRUD.base import BaseCRUD
import logging

logger = logging.getLogger(__name__)


class ServerEvaluationCRUD(BaseCRUD[ServerEvaluation]):
    """CRUD operations for ServerEvaluation model."""

    def __init__(self):
        super().__init__(ServerEvaluation)

    def create_evaluation(
        self,
        db: Session,
        run_id: int,
        round_number: int,
        metrics: Dict[str, Any],
        num_samples: Optional[int] = None,
    ) -> ServerEvaluation:
        """
        Create a server evaluation record from metrics dictionary.

        Args:
            db: Database session
            run_id: ID of the run
            round_number: Round number (1-indexed)
            metrics: Dictionary containing evaluation metrics
            num_samples: Number of samples evaluated

        Returns:
            Created ServerEvaluation instance
        """
        try:
            # Extract core metrics
            eval_data = {
                "run_id": run_id,
                "round_number": round_number,
                "loss": metrics.get("loss", 0.0),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score") or metrics.get("f1"),
                "auroc": metrics.get("auroc") or metrics.get("auc"),
                "num_samples": num_samples,
                "evaluation_time": datetime.now(),
            }

            # Extract confusion matrix if available (supports multiple formats)
            if "confusion_matrix" in metrics:
                # Nested dict format: {"confusion_matrix": {tp: ..., tn: ..., ...}}
                cm = metrics["confusion_matrix"]
                eval_data.update(
                    {
                        "true_positives": cm.get("true_positives") or cm.get("tp"),
                        "true_negatives": cm.get("true_negatives") or cm.get("tn"),
                        "false_positives": cm.get("false_positives") or cm.get("fp"),
                        "false_negatives": cm.get("false_negatives") or cm.get("fn"),
                    }
                )
            elif "server_cm_tp" in metrics:
                # Flat format: {"server_cm_tp": ..., "server_cm_tn": ..., ...}
                eval_data.update(
                    {
                        "true_positives": metrics.get("server_cm_tp"),
                        "true_negatives": metrics.get("server_cm_tn"),
                        "false_positives": metrics.get("server_cm_fp"),
                        "false_negatives": metrics.get("server_cm_fn"),
                    }
                )

            # Store any additional metrics
            excluded_keys = {
                "loss", "accuracy", "precision", "recall", "f1_score", "f1",
                "auroc", "auc", "confusion_matrix",
                "server_cm_tp", "server_cm_tn", "server_cm_fp", "server_cm_fn",
                "server_loss", "server_accuracy", "server_precision", "server_recall",
                "server_f1", "server_auroc"
            }
            additional = {
                k: v for k, v in metrics.items()
                if k not in excluded_keys
            }
            if additional:
                eval_data["additional_metrics"] = additional

            evaluation = self.create(db, **eval_data)
            logger.info(
                f"Created server evaluation for run_id={run_id}, round={round_number}"
            )
            return evaluation

        except Exception as e:
            logger.error(
                f"Failed to create server evaluation for run_id={run_id}, "
                f"round={round_number}: {e}"
            )
            raise

    def get_by_run(
        self, db: Session, run_id: int, order_by_round: bool = True
    ) -> List[ServerEvaluation]:
        """
        Get all server evaluations for a specific run.

        Args:
            db: Database session
            run_id: ID of the run
            order_by_round: Whether to order by round number (ascending)

        Returns:
            List of ServerEvaluation instances
        """
        query = db.query(ServerEvaluation).filter(ServerEvaluation.run_id == run_id)

        if order_by_round:
            query = query.order_by(ServerEvaluation.round_number)
        else:
            query = query.order_by(desc(ServerEvaluation.evaluation_time))

        return query.all()

    def get_by_round(
        self, db: Session, run_id: int, round_number: int
    ) -> Optional[ServerEvaluation]:
        """
        Get server evaluation for a specific run and round.

        Args:
            db: Database session
            run_id: ID of the run
            round_number: Round number

        Returns:
            ServerEvaluation instance or None
        """
        return (
            db.query(ServerEvaluation)
            .filter(
                ServerEvaluation.run_id == run_id,
                ServerEvaluation.round_number == round_number,
            )
            .first()
        )

    def get_latest(self, db: Session, run_id: int) -> Optional[ServerEvaluation]:
        """
        Get the latest server evaluation for a run.

        Args:
            db: Database session
            run_id: ID of the run

        Returns:
            Latest ServerEvaluation instance or None
        """
        return (
            db.query(ServerEvaluation)
            .filter(ServerEvaluation.run_id == run_id)
            .order_by(desc(ServerEvaluation.round_number))
            .first()
        )

    def get_best_by_metric(
        self, db: Session, run_id: int, metric_name: str = "accuracy"
    ) -> Optional[ServerEvaluation]:
        """
        Get the server evaluation with the best value for a specific metric.

        Args:
            db: Database session
            run_id: ID of the run
            metric_name: Name of the metric (accuracy, recall, f1_score, etc.)

        Returns:
            ServerEvaluation instance with best metric or None
        """
        if not hasattr(ServerEvaluation, metric_name):
            logger.warning(f"Metric '{metric_name}' not found in ServerEvaluation")
            return None

        return (
            db.query(ServerEvaluation)
            .filter(ServerEvaluation.run_id == run_id)
            .order_by(desc(getattr(ServerEvaluation, metric_name)))
            .first()
        )

    def get_summary_stats(self, db: Session, run_id: int) -> Dict[str, Any]:
        """
        Get summary statistics for server evaluations of a run.

        Args:
            db: Database session
            run_id: ID of the run

        Returns:
            Dictionary with summary statistics
        """
        evaluations = self.get_by_run(db, run_id)

        if not evaluations:
            return {}

        latest = evaluations[-1]

        # Find best values
        best_accuracy = max(
            (e for e in evaluations if e.accuracy),
            key=lambda x: x.accuracy,
            default=None,
        )
        best_recall = max(
            (e for e in evaluations if e.recall), key=lambda x: x.recall, default=None
        )
        best_precision = max(
            (e for e in evaluations if e.precision),
            key=lambda x: x.precision,
            default=None,
        )
        best_f1 = max(
            (e for e in evaluations if e.f1_score),
            key=lambda x: x.f1_score,
            default=None,
        )

        return {
            "total_rounds": len(evaluations),
            "latest_round": latest.round_number,
            "latest_metrics": {
                "loss": latest.loss,
                "accuracy": latest.accuracy,
                "precision": latest.precision,
                "recall": latest.recall,
                "f1_score": latest.f1_score,
                "auroc": latest.auroc,
            },
            "best_accuracy": {
                "value": best_accuracy.accuracy if best_accuracy else None,
                "round": best_accuracy.round_number if best_accuracy else None,
            },
            "best_recall": {
                "value": best_recall.recall if best_recall else None,
                "round": best_recall.round_number if best_recall else None,
            },
            "best_precision": {
                "value": best_precision.precision if best_precision else None,
                "round": best_precision.round_number if best_precision else None,
            },
            "best_f1_score": {
                "value": best_f1.f1_score if best_f1 else None,
                "round": best_f1.round_number if best_f1 else None,
            },
        }

    def delete_by_run(self, db: Session, run_id: int) -> int:
        """
        Delete all server evaluations for a specific run.

        Args:
            db: Database session
            run_id: ID of the run

        Returns:
            Number of deleted records
        """
        count = (
            db.query(ServerEvaluation)
            .filter(ServerEvaluation.run_id == run_id)
            .delete()
        )
        db.flush()
        logger.info(f"Deleted {count} server evaluations for run_id={run_id}")
        return count


# Create a singleton instance
server_evaluation_crud = ServerEvaluationCRUD()
