"""MetricsService for extracting, aggregating, and transforming run metrics.

This service encapsulates all metrics-related business logic including:
- Strategy-based metric extraction (federated vs centralized)
- Metrics aggregation across runs
- Run data transformation to analytics format
- Analytics summary generation

All read-heavy operations are cached via CacheProvider for performance.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)
from .cache import CacheProvider, cache_key
from .definitions import MetricsService as MetricsServiceProtocol
from .metric_extractors import (
    CentralizedMetricExtractor,
    FederatedMetricExtractor,
    MetricExtractor,
)
from .transformers import (
    calculate_summary_statistics,
    find_best_epoch,
    transform_run_to_results,
)

if TYPE_CHECKING:
    from federated_pneumonia_detection.src.boundary.models import Run


class MetricsService(MetricsServiceProtocol):
    """Service for retrieving and aggregating metrics with caching."""

    def __init__(
        self,
        *,
        cache: CacheProvider,
        run_crud_obj: Any = run_crud,
        run_metric_crud_obj: Any = run_metric_crud,
        server_evaluation_crud_obj: Any = server_evaluation_crud,
    ):
        """Initialize MetricsService with dependencies.

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

    def get_run_metrics(self, db: Session, run_id: int) -> dict[str, Any]:
        """Get all metrics for a specific run in analytics format.

        Args:
            db: Database session.
            run_id: Run identifier.

        Returns:
            Dictionary containing transformed metric data including:
            - experiment_id, status, final_metrics
            - training_history (list of epoch data)
            - total_epochs, metadata, confusion_matrix
        """
        key = cache_key("get_run_metrics", (run_id,), {})

        def _compute() -> dict[str, Any]:
            run = self._run_crud.get_with_metrics(db, run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found")

            persisted_stats = self._get_persisted_stats(db, run)
            return transform_run_to_results(run, persisted_stats)

        return self._cache.get_or_set(key, _compute)

    def get_analytics_summary(
        self, db: Session, *, filters: dict[str, Any]
    ) -> dict[str, Any]:
        """Get aggregated analytics summary with filters.

        Args:
            db: Database session.
            filters: Dictionary with optional keys:
                - status: Run status filter
                - training_mode: Training mode filter
                - days: Time window in days (from now)

        Returns:
            Dictionary containing:
            - total_runs, success_rate
            - centralized/federated statistics
            - top_runs (list)
        """
        # Parse filters with defaults
        status = filters.get("status", "completed")
        training_mode = filters.get("training_mode")
        days = filters.get("days")

        # Filter runs at database level
        runs = self._run_crud.get_by_status_and_mode(
            db, status=status, training_mode=training_mode
        )

        # Apply time filter if specified
        if days:
            from datetime import datetime

            cutoff_date = datetime.now() - timedelta(days=days)
            runs = [r for r in runs if r.start_time and r.start_time >= cutoff_date]

        if not runs:
            return self._create_empty_response()

        # Split by training mode
        centralized_runs = [r for r in runs if r.training_mode == "centralized"]
        federated_runs = [r for r in runs if r.training_mode == "federated"]

        # Calculate statistics for each mode
        centralized_stats = self._calculate_run_statistics(db, centralized_runs)
        federated_stats = self._calculate_run_statistics(db, federated_runs)

        # Extract run details for ranking
        all_run_details = [
            d for d in (self._extract_run_detail(db, r) for r in runs) if d is not None
        ]
        top_runs = self._get_top_runs(all_run_details, metric="best_accuracy", limit=10)

        # Calculate success rate (filtered runs / total runs with status)
        total_runs = len(runs)
        all_status_runs = self._run_crud.get_by_status(db, status)
        filtered_run_ratio = (
            total_runs / len(all_status_runs) if len(all_status_runs) > 0 else 0.0
        )

        return {
            "total_runs": total_runs,
            "success_rate": round(filtered_run_ratio, 4),
            "centralized": centralized_stats,
            "federated": federated_stats,
            "top_runs": top_runs,
        }

    def get_run_detail(self, db: Session, run_id: int) -> dict[str, Any]:
        """Get detailed metrics for a specific run.

        Args:
            db: Database session.
            run_id: Run identifier.

        Returns:
            Dictionary containing:
            - run_id, training_mode, best_accuracy
            - best_precision, best_recall, best_f1
            - duration_minutes, start_time, status
        """
        run = self._run_crud.get_with_metrics(db, run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        detail = self._extract_run_detail(db, run)
        if detail is None:
            raise ValueError(f"No metrics found for run {run_id}")

        return detail

    # Internal helper methods

    def _get_persisted_stats(self, db: Session, run: Run) -> Optional[dict[str, float]]:
        """Get persisted final epoch stats for a run.

        Args:
             db: Database session.
             run: Run object to fetch stats for.

        Returns:
             Dictionary with final epoch stats or None if not available.
        """
        if run.training_mode == "federated":  # type: ignore[misc]
            # Get stats from server evaluation additional_metrics
            latest_eval = self._server_evaluation_crud.get_latest(db, run.id)
            if (
                latest_eval
                and latest_eval.additional_metrics
                and "final_epoch_stats" in latest_eval.additional_metrics
            ):
                return latest_eval.additional_metrics["final_epoch_stats"]
        else:
            # Get stats from run_metrics with "final_" prefix
            final_metrics = self._run_metric_crud.get_by_run(db, run.id)
            if final_metrics:
                # Build stats dict from final_* metrics
                stats_dict = {}
                for m in final_metrics:
                    if m.metric_name.startswith("final_"):
                        stat_name = m.metric_name.replace("final_", "")
                        stats_dict[stat_name] = m.metric_value
                return stats_dict if stats_dict else None

        return None

    def _calculate_run_statistics(self, db: Session, runs: list[Run]) -> dict[str, Any]:
        """Calculate aggregated statistics for a list of runs.

        Args:
            db: Database session.
            runs: List of Run objects to analyze.

        Returns:
            Dictionary with aggregated statistics:
            - count, avg_accuracy, avg_precision
            - avg_recall, avg_f1, avg_duration_minutes
        """
        if not runs:
            return {
                "count": 0,
                "avg_accuracy": None,
                "avg_precision": None,
                "avg_recall": None,
                "avg_f1": None,
                "avg_duration_minutes": None,
            }

        accuracies: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        f1_scores: list[float] = []
        durations: list[float] = []

        for run in runs:
            extractor = self._get_metric_extractor(run)
            if acc := extractor.get_best_metric(db, run.id, "accuracy"):  # type: ignore[arg-type]
                accuracies.append(acc)
            if prec := extractor.get_best_metric(db, run.id, "precision"):  # type: ignore[arg-type]
                precisions.append(prec)
            if rec := extractor.get_best_metric(db, run.id, "recall"):  # type: ignore[arg-type]
                recalls.append(rec)
            if f1 := extractor.get_best_metric(db, run.id, "f1_score"):  # type: ignore[arg-type]
                f1_scores.append(f1)

            if run.start_time and run.end_time:  # type: ignore[misc]
                duration = (run.end_time - run.start_time).total_seconds() / 60
                durations.append(duration)

        return {
            "count": len(runs),
            "avg_accuracy": self._safe_average(accuracies),
            "avg_precision": self._safe_average(precisions),
            "avg_recall": self._safe_average(recalls),
            "avg_f1": self._safe_average(f1_scores),
            "avg_duration_minutes": self._safe_average(durations),
        }

    def _extract_run_detail(self, db: Session, run: Run) -> Optional[dict[str, Any]]:
        """Extract detailed metrics for a single run.

        Args:
            db: Database session.
            run: Run object to extract metrics from.

        Returns:
            Dictionary with run details or None if metrics unavailable.
        """
        try:
            extractor = self._get_metric_extractor(run)
            accuracy = extractor.get_best_metric(db, run.id, "accuracy")  # type: ignore[arg-type]

            if accuracy is None:
                return None

            duration_minutes = None
            if run.start_time and run.end_time:  # type: ignore[misc]
                duration_minutes = round(
                    (run.end_time - run.start_time).total_seconds() / 60,
                    2,
                )

            return {
                "run_id": run.id,
                "training_mode": run.training_mode,
                "best_accuracy": round(accuracy, 4) if accuracy else None,
                "best_precision": round(
                    extractor.get_best_metric(db, run.id, "precision") or 0,  # type: ignore[arg-type]
                    4,
                ),
                "best_recall": round(
                    extractor.get_best_metric(db, run.id, "recall") or 0,  # type: ignore[arg-type]
                    4,
                ),
                "best_f1": round(
                    extractor.get_best_metric(db, run.id, "f1_score") or 0,  # type: ignore[arg-type]
                    4,
                ),
                "duration_minutes": duration_minutes,
                "start_time": run.start_time.isoformat() if run.start_time else None,  # type: ignore[misc]
                "status": run.status,
            }
        except Exception:
            return None

    def _get_top_runs(
        self,
        runs: list[dict[str, Any]],
        *,
        metric: str = "best_accuracy",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top-N runs sorted by metric descending.

        Args:
            runs: List of run detail dictionaries.
            metric: Metric key to sort by.
            limit: Maximum number of runs to return.

        Returns:
            Top N runs sorted by metric descending.
        """
        return sorted(runs, key=lambda x: x.get(metric, 0.0), reverse=True)[:limit]

    def _safe_average(self, values: list[float]) -> Optional[float]:
        """Calculate average of values, handling empty lists gracefully.

        Args:
            values: List of numeric values.

        Returns:
            Average rounded to 4 decimals or None if list is empty.
        """
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    def _get_metric_extractor(self, run: Run) -> MetricExtractor:
        """Factory method to get appropriate metric extractor for run type.

        Args:
             run: Run object.

        Returns:
             MetricExtractor instance (Federated or Centralized).
        """
        if run.training_mode == "federated":  # type: ignore[misc]
            return FederatedMetricExtractor(self._server_evaluation_crud)
        return CentralizedMetricExtractor(self._run_metric_crud)

    def _create_empty_response(self) -> dict[str, Any]:
        """Generate empty analytics response when no runs found.

        Returns:
            Dictionary with zeroed/null values.
        """
        empty_stats = {
            "count": 0,
            "avg_accuracy": None,
            "avg_precision": None,
            "avg_recall": None,
            "avg_f1": None,
            "avg_duration_minutes": None,
        }
        return {
            "total_runs": 0,
            "success_rate": 0.0,
            "centralized": empty_stats,
            "federated": empty_stats,
            "top_runs": [],
        }
