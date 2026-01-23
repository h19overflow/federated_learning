"""Shared run summary construction logic for centralized and federated modes."""

from typing import Any, Dict, Optional

from sqlalchemy import desc

from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from federated_pneumonia_detection.src.boundary.models import (
    Run,
    RunMetric,
    ServerEvaluation,
)
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

logger = get_logger(__name__)


class RunSummaryBuilder:
    """Builds run summaries with base fields and mode-specific enhancements."""

    @staticmethod
    def build(run: Run, db) -> Dict[str, Any]:
        """Build complete run summary (base + mode-specific)."""
        summary = RunSummaryBuilder._build_base_summary(run)
        if run.training_mode == "federated":
            summary["federated_info"] = FederatedRunSummarizer.summarize(run, db)
        else:
            summary["federated_info"] = None
        summary["final_epoch_stats"] = RunSummaryBuilder._get_final_epoch_stats(run, db)
        return summary

    @staticmethod
    def _build_base_summary(run: Run) -> Dict[str, Any]:
        """Build base summary fields common across all training modes."""
        best_val_recall = 0.0
        best_val_accuracy = 0.0
        if run.metrics:
            recall_vals = [
                m.metric_value for m in run.metrics if m.metric_name == "val_recall"
            ]
            if recall_vals:
                best_val_recall = max(recall_vals)
            acc_vals = [
                m.metric_value for m in run.metrics if m.metric_name == "val_acc"
            ]
            if acc_vals:
                best_val_accuracy = max(acc_vals)

        return {
            "id": run.id,
            "training_mode": run.training_mode,
            "status": run.status,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "best_val_recall": best_val_recall,
            "best_val_accuracy": best_val_accuracy,
            "metrics_count": len(run.metrics) if hasattr(run, "metrics") else 0,
            "run_description": run.run_description,
        }

    @staticmethod
    def _get_final_epoch_stats(run: Run, db) -> Optional[Dict[str, float]]:
        """Retrieve persisted final epoch stats for a run."""
        if run.training_mode == "centralized":
            # Query RunMetric for 'final_*' metrics
            final_metrics = (
                db.query(RunMetric)
                .filter(
                    RunMetric.run_id == run.id,
                    RunMetric.metric_name.in_(
                        [
                            "final_sensitivity",
                            "final_specificity",
                            "final_precision_cm",
                            "final_accuracy_cm",
                            "final_f1_cm",
                        ]
                    ),
                )
                .all()
            )

            if len(final_metrics) == 5:
                return {
                    m.metric_name.replace("final_", ""): m.metric_value
                    for m in final_metrics
                }

        elif run.training_mode == "federated":
            # Query last ServerEvaluation.additional_metrics
            last_eval = (
                db.query(ServerEvaluation)
                .filter(ServerEvaluation.run_id == run.id)
                .order_by(desc(ServerEvaluation.round_number))
                .first()
            )

            if last_eval and last_eval.additional_metrics:
                return last_eval.additional_metrics.get("final_epoch_stats")

        return None


class FederatedRunSummarizer:
    """Builds federated-specific summary from server evaluations."""

    @staticmethod
    def summarize(run: Run, db) -> Dict[str, Any]:
        """Build federated summary excluding round 0 initial evaluation."""
        server_evals = server_evaluation_crud.get_by_run(db, run.id)
        num_clients = len(run.clients) if run.clients else 0
        training_rounds = [e for e in server_evals if e.round_number > 0]

        logger.info(
            f"[FedSummarizer] Run {run.id}: {len(server_evals)} evals "
            f"({len(training_rounds)} training), {num_clients} clients",
        )

        info = {
            "num_rounds": len(training_rounds),
            "num_clients": num_clients,
            "has_server_evaluation": len(training_rounds) > 0,
        }

        if training_rounds:
            info["best_accuracy"] = max(
                (e.accuracy for e in training_rounds if e.accuracy),
                default=None,
            )
            info["best_recall"] = max(
                (e.recall for e in training_rounds if e.recall),
                default=None,
            )
            latest = training_rounds[-1]
            info["latest_round"] = latest.round_number
            info["latest_accuracy"] = latest.accuracy

            logger.info(
                f"[FedSummarizer] Run {run.id}: best_acc={info['best_accuracy']}, "
                f"latest_acc={latest.accuracy}, latest_rnd={latest.round_number}",
            )
        else:
            logger.warning(f"[FedSummarizer] Run {run.id}: No training evaluations!")

        return info
