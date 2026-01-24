"""Metric extractor for federated training runs."""

from typing import Optional

from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    server_evaluation_crud,
)

from .base import MetricExtractor


class FederatedMetricExtractor(MetricExtractor):
    """Extract metrics from federated runs via server evaluations."""

    def __init__(self, crud=None):
        """Initialize with optional CRUD dependency injection.

        Args:
            crud: ServerEvaluationCRUD instance (defaults to singleton).
        """
        self._crud = crud or server_evaluation_crud

    def get_best_metric(
        self,
        db: Session,
        run_id: int,
        metric_name: str,
    ) -> Optional[float]:
        """Get best metric from server evaluations, excluding round 0.

        Args:
            db: Database session.
            run_id: Run identifier.
            metric_name: Metric name (e.g., 'accuracy', 'precision').

        Returns:
            Best metric value or None if unavailable.
        """
        summary = self._crud.get_summary_stats(db, run_id)
        if not summary:
            return None
        metric_key = f"best_{metric_name}"
        if metric_key not in summary or not summary[metric_key]:
            return None
        return summary[metric_key].get("value")
