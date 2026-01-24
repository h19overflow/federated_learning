"""Metric extractor for centralized training runs."""

from typing import Optional

from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud

from .base import MetricExtractor


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

    def __init__(self, crud=None):
        """Initialize with optional CRUD dependency injection.

        Args:
            crud: RunMetricCRUD instance (defaults to singleton).
        """
        self._crud = crud or run_metric_crud

    def get_best_metric(
        self,
        db: Session,
        run_id: int,
        metric_name: str,
    ) -> Optional[float]:
        """Get best metric from run_metrics table.

        Args:
            db: Database session.
            run_id: Run identifier.
            metric_name: Metric name (e.g., 'accuracy', 'precision').

        Returns:
            Best metric value or None if unavailable.
        """
        # Use mapping to get actual database field name
        metric_field = self.METRIC_NAME_MAP.get(metric_name, f"val_{metric_name}")
        best = self._crud.get_best_metric(db, run_id, metric_field, maximize=True)
        return best.metric_value if best else None  # type: ignore[return-value]
