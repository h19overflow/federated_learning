from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from federated_pneumonia_detection.src.boundary.CRUD.base import BaseCRUD
from federated_pneumonia_detection.src.boundary.models import RunMetric

# TODO: run metrics are still not being recordede for the federated training mode
class RunMetricCRUD(BaseCRUD[RunMetric]):
    """CRUD operations for RunMetric model."""

    def __init__(self):
        super().__init__(RunMetric)

    def get_by_run(self, db: Session, run_id: int) -> List[RunMetric]:
        """Get all metrics for a specific run."""
        return db.query(self.model).filter(self.model.run_id == run_id).all()

    def get_by_metric_name(
        self, db: Session, run_id: int, metric_name: str
    ) -> List[RunMetric]:
        """Get specific metric for a run."""
        return (
            db.query(self.model)
            .filter(self.model.run_id == run_id, self.model.metric_name == metric_name)
            .order_by(self.model.step)
            .all()
        )

    def get_by_dataset_type(
        self, db: Session, run_id: int, dataset_type: str
    ) -> List[RunMetric]:
        """Get metrics for a specific dataset type."""
        return (
            db.query(self.model)
            .filter(
                self.model.run_id == run_id, self.model.dataset_type == dataset_type
            )
            .all()
        )

    def get_latest_by_metric(
        self, db: Session, run_id: int, metric_name: str
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
            .filter(self.model.run_id == run_id, self.model.step == step)
            .all()
        )

    def get_best_metric(
        self, db: Session, run_id: int, metric_name: str, maximize: bool = True
    ) -> Optional[RunMetric]:
        """Get the best value for a metric (max or min)."""
        query = db.query(self.model).filter(
            self.model.run_id == run_id, self.model.metric_name == metric_name
        )

        if maximize:
            return query.order_by(self.model.metric_value.desc()).first()
        else:
            return query.order_by(self.model.metric_value.asc()).first()

    def get_metric_stats(
        self, db: Session, run_id: int, metric_name: str
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
        self, db: Session, run_id: int, metric_name: str, dataset_type: str
    ) -> List[RunMetric]:
        """Get metrics filtered by name and dataset type."""
        return (
            db.query(self.model)
            .filter(
                self.model.run_id == run_id,
                self.model.metric_name == metric_name,
                self.model.dataset_type == dataset_type,
            )
            .order_by(self.model.step)
            .all()
        )


run_metric_crud = RunMetricCRUD()
