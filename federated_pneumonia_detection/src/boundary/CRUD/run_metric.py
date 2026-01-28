import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session, joinedload

from federated_pneumonia_detection.src.boundary.CRUD.base import BaseCRUD
from federated_pneumonia_detection.src.boundary.models import RunMetric

logger = logging.getLogger(__name__)


class RunMetricCRUD(BaseCRUD[RunMetric]):
    """CRUD operations for RunMetric model."""

    def __init__(self):
        super().__init__(RunMetric)

    def get_by_run(self, db: Session, run_id: int) -> List[RunMetric]:
        """Get all metrics for a specific run."""
        return (
            db.query(self.model)
            .options(
                joinedload(self.model.run),
                joinedload(self.model.client),
                joinedload(self.model.round),
            )
            .filter(self.model.run_id == run_id)
            .all()
        )

    def get_by_metric_name(
        self,
        db: Session,
        run_id: int,
        metric_name: str,
    ) -> List[RunMetric]:
        """Get specific metric for a run."""
        return (
            db.query(self.model)
            .options(
                joinedload(self.model.run),
                joinedload(self.model.client),
                joinedload(self.model.round),
            )
            .filter(self.model.run_id == run_id, self.model.metric_name == metric_name)
            .order_by(self.model.step)
            .all()
        )

    def get_by_dataset_type(
        self,
        db: Session,
        run_id: int,
        dataset_type: str,
    ) -> List[RunMetric]:
        """Get metrics for a specific dataset type."""
        return (
            db.query(self.model)
            .options(
                joinedload(self.model.run),
                joinedload(self.model.client),
                joinedload(self.model.round),
            )
            .filter(
                self.model.run_id == run_id,
                self.model.dataset_type == dataset_type,
            )
            .all()
        )

    def get_latest_by_metric(
        self,
        db: Session,
        run_id: int,
        metric_name: str,
    ) -> Optional[RunMetric]:
        """Get the latest value for a specific metric."""
        return (
            db.query(self.model)
            .filter(self.model.run_id == run_id, self.model.metric_name == metric_name)
            .order_by(self.model.step.desc())
            .first()
        )

    def get_by_step(self, db: Session, run_id: int, step: int) -> List[RunMetric]:
        """Get all metrics at a specific step."""
        return (
            db.query(self.model)
            .options(
                joinedload(self.model.run),
                joinedload(self.model.client),
                joinedload(self.model.round),
            )
            .filter(self.model.run_id == run_id, self.model.step == step)
            .all()
        )

    def get_best_metric(
        self,
        db: Session,
        run_id: int,
        metric_name: str,
        maximize: bool = True,
    ) -> Optional[RunMetric]:
        """Get the best value for a metric (max or min)."""
        query = db.query(self.model).filter(
            self.model.run_id == run_id,
            self.model.metric_name == metric_name,
        )

        if maximize:
            return query.order_by(self.model.metric_value.desc()).first()
        else:
            return query.order_by(self.model.metric_value.asc()).first()

    def get_metric_stats(
        self,
        db: Session,
        run_id: int,
        metric_name: str,
    ) -> Dict[str, Any]:
        """Get statistics for a specific metric."""
        metrics = self.get_by_metric_name(db, run_id, metric_name)

        if not metrics:
            return {}

        values = [m.metric_value for m in metrics]
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "count": len(values),
            "latest": metrics[-1].metric_value if metrics else None,
        }

    def get_metrics_by_name_and_dataset(
        self,
        db: Session,
        run_id: int,
        metric_name: str,
        dataset_type: str,
    ) -> List[RunMetric]:
        """Get metrics filtered by name and dataset type."""
        return (
            db.query(self.model)
            .options(
                joinedload(self.model.run),
                joinedload(self.model.client),
                joinedload(self.model.round),
            )
            .filter(
                self.model.run_id == run_id,
                self.model.metric_name == metric_name,
                self.model.dataset_type == dataset_type,
            )
            .order_by(self.model.step)
            .all()
        )

    def get_by_run_grouped_by_client(
        self,
        db: Session,
        run_id: int,
    ) -> Dict[int, List[RunMetric]]:
        """Get metrics for a federated run grouped by client_id.

        Args:
            db: Database session.
            run_id: Run ID to fetch metrics for.

        Returns:
            Dictionary mapping client_id to list of RunMetric objects.
            Only includes metrics with non-null client_id.
        """
        metrics = (
            db.query(self.model)
            .options(
                joinedload(self.model.client),
                joinedload(self.model.round),
            )
            .filter(
                self.model.run_id == run_id,
                self.model.client_id.isnot(None),
            )
            .order_by(self.model.client_id, self.model.step)
            .all()
        )

        grouped: Dict[int, List[RunMetric]] = {}
        for metric in metrics:
            client_id = metric.client_id
            if client_id not in grouped:
                grouped[client_id] = []
            grouped[client_id].append(metric)

        return grouped

    def get_aggregated_metrics_by_run(
        self,
        db: Session,
        run_id: int,
    ) -> List[RunMetric]:
        """Get aggregated (server-side) metrics for a federated run.

        Args:
            db: Database session.
            run_id: Run ID to fetch metrics for.

        Returns:
            List of RunMetric objects with context='aggregated'.
        """
        return (
            db.query(self.model)
            .filter(
                self.model.run_id == run_id,
                self.model.context == "aggregated",
            )
            .order_by(self.model.step)
            .all()
        )

    def create_final_epoch_stats(
        self,
        db: Session,
        run_id: int,
        stats_dict: Dict[str, float],
        final_epoch: int,
    ) -> List[RunMetric]:
        """
        Persist final epoch confusion matrix statistics as RunMetric rows.

        Args:
            db: Database session
            run_id: Run ID
            stats_dict: Dict with keys: sensitivity, specificity, precision_cm, accuracy_cm, f1_cm  # noqa: E501
            final_epoch: Final epoch number (step value)

        Returns:
            List of created RunMetric instances
        """
        # Map stat keys to metric names
        metric_mapping = {
            "sensitivity": "final_sensitivity",
            "specificity": "final_specificity",
            "precision_cm": "final_precision_cm",
            "accuracy_cm": "final_accuracy_cm",
            "f1_cm": "final_f1_cm",
        }

        # Prepare bulk create data
        metrics_data = []
        for stat_key, metric_name in metric_mapping.items():
            if stat_key in stats_dict:
                metrics_data.append(
                    {
                        "run_id": run_id,
                        "metric_name": metric_name,
                        "metric_value": stats_dict[stat_key],
                        "step": final_epoch,
                        "dataset_type": "validation",
                        "context": "final_epoch",
                    }
                )

        # Create metrics using bulk create for efficiency
        created_metrics = self.bulk_create(db, metrics_data)

        logger.info(
            f"Created {len(created_metrics)} final epoch stats for run_id={run_id}, "
            f"final_epoch={final_epoch}"
        )

        return created_metrics


run_metric_crud = RunMetricCRUD()
