"""SummaryService for building and caching run summaries.

This service encapsulates run summary construction logic including:
- Base summary fields (id, mode, status, metrics)
- Mode-specific enhancements (federated info, final epoch stats)
- Caching of expensive summary operations
- List and individual run summary retrieval
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from federated_pneumonia_detection.src.boundary.models import (
    RunMetric,
    ServerEvaluation,
)
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from ..infrastructure import CacheProvider, cache_key

if TYPE_CHECKING:
    from federated_pneumonia_detection.src.boundary.models import Run

logger = get_logger(__name__)


class SummaryService:
    """Service for building and retrieving run summaries with caching."""

    def __init__(
        self,
        *,
        cache: CacheProvider,
        run_crud_obj: Any = run_crud,
        run_metric_crud_obj: Any = run_metric_crud,
        server_evaluation_crud_obj: Any = server_evaluation_crud,
    ):
        """Initialize SummaryService with dependencies.

        Args:
            cache: CacheProvider instance for caching expensive operations.
            run_crud_obj: RunCRUD instance (injected for testability).
            run_metric_crud_obj: RunMetricCRUD instance.
            server_evaluation_crud_obj: ServerEvaluationCRUD instance.
        """
        self._cache = cache
        self._run_crud = run_crud_obj
        self._run_metric_crud = run_metric_crud_obj
        self._server_evaluation_crud = server_evaluation_crud_obj

    def get_run_summary(self, db: Session, run_id: int) -> dict[str, Any]:
        """Get summary for a specific run with caching.

        Args:
            db: Database session.
            run_id: Run identifier.

        Returns:
            Dictionary containing:
            - id, training_mode, status
            - start_time, end_time
            - best_val_recall, best_val_accuracy
            - metrics_count, run_description
            - federated_info (if federated mode)
            - final_epoch_stats
        """
        key = cache_key("get_run_summary", (run_id,), {})

        def _compute() -> dict[str, Any]:
            run = self._run_crud.get_with_metrics(db, run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found")
            return self._build_run_summary(run, db)

        return self._cache.get_or_set(key, _compute)

    def list_run_summaries(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Get list of run summaries with optional filtering and caching.

        Args:
            db: Database session.
            skip: Number of runs to skip.
            limit: Maximum number of runs to return.
            filters: Optional dictionary with keys:
                - status: Run status filter
                - training_mode: Training mode filter

        Returns:
            Dictionary containing:
            - total: Total number of runs matching filters
            - summaries: List of run summary dictionaries
        """
        filters = filters or {}
        key = cache_key("list_run_summaries", (skip, limit), filters)

        def _compute() -> dict[str, Any]:
            status = filters.get("status")
            training_mode = filters.get("training_mode")

            # Get filtered runs
            if status and training_mode:
                runs = self._run_crud.get_by_status_and_mode(
                    db, status=status, training_mode=training_mode
                )
            elif status:
                runs = self._run_crud.get_by_status(db, status)
            elif training_mode:
                runs = self._run_crud.get_by_training_mode(db, training_mode)
            else:
                runs = self._run_crud.get_all(db)

            total = len(runs)
            paginated_runs = runs[skip : skip + limit]

            summaries = []
            for run in paginated_runs:
                try:
                    summary = self._build_run_summary(run, db)
                    summaries.append(summary)
                except Exception as e:
                    logger.warning(f"Failed to build summary for run {run.id}: {e}")

            return {"total": total, "summaries": summaries}

        return self._cache.get_or_set(key, _compute)

    def list_runs_with_summaries(
        self,
        db: Session,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None,
        training_mode: Optional[str] = None,
        sort_by: str = "start_time",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        """Get paginated list of run summaries with filtering and sorting.

        This method handles all the complexity of fetching, filtering, sorting,
        and building summaries for the API layer.

        Args:
            db: Database session.
            limit: Maximum number of runs to return (1-1000).
            offset: Number of runs to skip for pagination.
            status: Filter by run status (optional).
            training_mode: Filter by training mode (optional).
            sort_by: Field to sort by (default: "start_time").
            sort_order: Sort direction "asc" or "desc" (default: "desc").

        Returns:
            Dictionary containing:
            - runs: List of run summary dictionaries
            - total: Total number of runs matching filters
        """
        # Use CRUD method for filtered, sorted, paginated list
        runs, total_count = self._run_crud.list_with_filters(
            db,
            status=status,
            training_mode=training_mode,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        )

        # Build summaries for all runs
        summaries = []
        for run in runs:
            try:
                summary = self._build_run_summary(run, db)
                summaries.append(summary)
            except Exception as e:
                logger.error(
                    f"Failed to build summary for run {run.id}: {e}",
                    exc_info=True,
                )
                # Graceful degradation: append minimal summary with error indicator
                summaries.append(
                    {
                        "id": run.id,
                        "training_mode": run.training_mode,
                        "status": run.status,
                        "start_time": run.start_time.isoformat()
                        if run.start_time
                        else None,
                        "end_time": run.end_time.isoformat()
                        if run.end_time
                        else None,
                        "best_val_recall": 0.0,
                        "best_val_accuracy": 0.0,
                        "metrics_count": 0,
                        "run_description": None,
                        "federated_info": None,
                        "final_epoch_stats": None,
                        "error": "Summary unavailable",
                    }
                )

        return {"runs": summaries, "total": total_count}

    # Internal helper methods

    def _build_run_summary(self, run: Run, db: Session) -> dict[str, Any]:
        """Build complete run summary (base + mode-specific).

        Args:
            run: Run object to summarize.
            db: Database session.

        Returns:
            Dictionary with complete run summary.
        """
        summary = self._build_base_summary(run)

        # Add mode-specific info
        if run.training_mode == "federated":  # type: ignore[misc]
            summary["federated_info"] = self._build_federated_info(run, db)
        else:
            summary["federated_info"] = None

        # Add final epoch stats
        summary["final_epoch_stats"] = self._get_final_epoch_stats(run, db)

        return summary

    def _build_base_summary(self, run: Run) -> dict[str, Any]:
        """Build base summary fields common across all training modes.

        Args:
            run: Run object to summarize.

        Returns:
            Dictionary with base summary fields.
        """
        best_val_recall = 0.0
        best_val_accuracy = 0.0

        if run.metrics:
            recall_vals = [
                m.metric_value
                for m in run.metrics
                if m.metric_name == "val_recall"  # type: ignore[misc]
            ]
            if recall_vals:
                best_val_recall = max(recall_vals)

            acc_vals = [
                m.metric_value
                for m in run.metrics
                if m.metric_name == "val_acc"  # type: ignore[misc]
            ]
            if acc_vals:
                best_val_accuracy = max(acc_vals)

        return {
            "id": run.id,
            "training_mode": run.training_mode,
            "status": run.status,
            "start_time": run.start_time.isoformat() if run.start_time else None,  # type: ignore[misc]
            "end_time": run.end_time.isoformat() if run.end_time else None,  # type: ignore[misc]
            "best_val_recall": best_val_recall,
            "best_val_accuracy": best_val_accuracy,
            "metrics_count": len(run.metrics) if hasattr(run, "metrics") else 0,
            "run_description": run.run_description,
        }

    def _build_federated_info(self, run: Run, db: Session) -> dict[str, Any]:
        """Build federated-specific summary from server evaluations.

        Args:
             run: Run object (must be federated mode).
             db: Database session.

        Returns:
             Dictionary with federated summary info.
        """
        server_evals = self._server_evaluation_crud.get_by_run(db, run.id)
        num_clients = len(run.clients) if run.clients else 0
        training_rounds = [e for e in server_evals if e.round_number > 0]  # type: ignore[misc]

        logger.info(
            f"[SummarySvc] Run {run.id}: {len(server_evals)} evals "
            f"({len(training_rounds)} training), {num_clients} clients",
        )

        info = {
            "num_rounds": len(training_rounds),
            "num_clients": num_clients,
            "has_server_evaluation": len(training_rounds) > 0,
        }

        if training_rounds:
            # Get best accuracy, defaulting to 0.0 if no valid values
            best_acc_values = [
                e.accuracy for e in training_rounds if e.accuracy is not None
            ]
            info["best_accuracy"] = max(best_acc_values) if best_acc_values else 0.0

            # Get best recall, defaulting to 0.0 if no valid values
            best_recall_values = [
                e.recall for e in training_rounds if e.recall is not None
            ]
            info["best_recall"] = max(best_recall_values) if best_recall_values else 0.0

            latest = training_rounds[-1]
            info["latest_round"] = latest.round_number
            # Ensure latest_accuracy is not None before using
            latest_accuracy = latest.accuracy if latest.accuracy is not None else 0.0
            info["latest_accuracy"] = latest_accuracy

            logger.info(
                f"[SummarySvc] Run {run.id}: best_acc={info['best_accuracy']}, "
                f"latest_acc={latest_accuracy}, latest_rnd={latest.round_number}",
            )
        else:
            logger.warning(f"[SummarySvc] Run {run.id}: No training evaluations!")

        return info

    def _get_final_epoch_stats(
        self, run: Run, db: Session
    ) -> Optional[dict[str, float]]:
        """Retrieve persisted final epoch stats for a run.

        Args:
            run: Run object to fetch stats for.
            db: Database session.

        Returns:
            Dictionary with final epoch stats or None if not available.
        """
        if run.training_mode == "centralized":  # type: ignore[misc]
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
                return {  # type: ignore[return-value]
                    m.metric_name.replace("final_", ""): m.metric_value
                    for m in final_metrics
                }

        elif run.training_mode == "federated":  # type: ignore[misc]
            # Query last ServerEvaluation.additional_metrics
            last_eval = (
                db.query(ServerEvaluation)
                .filter(ServerEvaluation.run_id == run.id)
                .order_by(desc(ServerEvaluation.round_number))
                .first()
            )

            if last_eval and last_eval.additional_metrics:  # type: ignore[misc]
                # .get() may return None, so ensure we return None explicitly if not found  # noqa: E501
                final_stats = last_eval.additional_metrics.get("final_epoch_stats")
                if final_stats is not None:
                    return final_stats

        return None
