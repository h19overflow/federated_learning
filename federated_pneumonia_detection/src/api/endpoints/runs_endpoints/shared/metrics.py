"""
Shared metric extraction and aggregation logic for run endpoints.

Provides strategy-based metric extractors for both centralized and federated
training runs. Follows SOLID principles with DIP and Strategy pattern.
"""

from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.engine import Run
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)


class MetricExtractor(ABC):
    """Base class for extracting metrics from different training modes."""

    @abstractmethod
    def get_best_metric(self, db: Session, run_id: int, metric_name: str) -> Optional[float]:
        """Extract best metric value for a run."""
        pass


class FederatedMetricExtractor(MetricExtractor):
    """Extract metrics from federated runs via server evaluations."""

    def get_best_metric(self, db: Session, run_id: int, metric_name: str) -> Optional[float]:
        """Get best metric from server evaluations, excluding round 0."""
        summary = server_evaluation_crud.get_summary_stats(db, run_id)
        if not summary:
            return None
        metric_key = f"best_{metric_name}"
        if metric_key not in summary or not summary[metric_key]:
            return None
        return summary[metric_key].get("value")


class CentralizedMetricExtractor(MetricExtractor):
    """Extract metrics from centralized runs via run metrics."""

    # Map standardized API metric names to database field names
    METRIC_NAME_MAP = {
        "accuracy": "val_acc",
        "precision": "val_precision",
        "recall": "val_recall",
        "f1_score": "val_f1",
        "auroc": "val_auroc",
    }

    def get_best_metric(self, db: Session, run_id: int, metric_name: str) -> Optional[float]:
        """Get best metric from run_metrics table."""
        # Use mapping to get actual database field name
        metric_field = self.METRIC_NAME_MAP.get(metric_name, f"val_{metric_name}")
        best = run_metric_crud.get_best_metric(db, run_id, metric_field, maximize=True)
        return best.metric_value if best else None


def get_metric_extractor(run: Run) -> MetricExtractor:
    """Factory function to get appropriate metric extractor for run type."""
    if run.training_mode == "federated":
        return FederatedMetricExtractor()
    return CentralizedMetricExtractor()


class MetricsAggregator:
    """Aggregates metrics from collections of runs or evaluations."""

    @staticmethod
    def get_best_metric(
        items: List[Any], field: str, default: Optional[float] = None
    ) -> Optional[float]:
        """Get maximum value from collection."""
        values = [getattr(item, field) for item in items if getattr(item, field) is not None]
        return max(values) if values else default

    @staticmethod
    def get_worst_metric(
        items: List[Any], field: str, default: Optional[float] = None
    ) -> Optional[float]:
        """Get minimum value from collection."""
        values = [getattr(item, field) for item in items if getattr(item, field) is not None]
        return min(values) if values else default

    @staticmethod
    def get_latest_value(items: List[Any], field: str) -> Optional[float]:
        """Get value from last item in collection."""
        if not items:
            return None
        return getattr(items[-1], field, None)

    @staticmethod
    def calculate_best_validation_recall(metrics: List[Any]) -> float:
        """Calculate best validation recall from metric collection."""
        val_recalls = [m.metric_value for m in metrics if m.metric_name == "val_recall"]
        return max(val_recalls) if val_recalls else 0.0


class RunAggregator:
    """Aggregates statistics across multiple runs."""

    @staticmethod
    def calculate_statistics(db: Session, runs: List[Run]) -> Dict[str, Any]:
        """Calculate aggregated statistics for runs (centralized/federated)."""
        if not runs:
            return {
                "count": 0,
                "avg_accuracy": None,
                "avg_precision": None,
                "avg_recall": None,
                "avg_f1": None,
                "avg_duration_minutes": None,
            }

        accuracies: List[float] = []
        precisions: List[float] = []
        recalls: List[float] = []
        f1_scores: List[float] = []
        durations: List[float] = []

        for run in runs:
            extractor = get_metric_extractor(run)
            if acc := extractor.get_best_metric(db, run.id, "accuracy"):
                accuracies.append(acc)
            if prec := extractor.get_best_metric(db, run.id, "precision"):
                precisions.append(prec)
            if rec := extractor.get_best_metric(db, run.id, "recall"):
                recalls.append(rec)
            if f1 := extractor.get_best_metric(db, run.id, "f1_score"):
                f1_scores.append(f1)

            if run.start_time and run.end_time:
                duration = (run.end_time - run.start_time).total_seconds() / 60
                durations.append(duration)

        return {
            "count": len(runs),
            "avg_accuracy": RunAggregator._safe_average(accuracies),
            "avg_precision": RunAggregator._safe_average(precisions),
            "avg_recall": RunAggregator._safe_average(recalls),
            "avg_f1": RunAggregator._safe_average(f1_scores),
            "avg_duration_minutes": RunAggregator._safe_average(durations),
        }

    @staticmethod
    def _safe_average(values: List[float]) -> Optional[float]:
        """Calculate average of values, handling empty lists gracefully."""
        if not values:
            return None
        return round(sum(values) / len(values), 4)
