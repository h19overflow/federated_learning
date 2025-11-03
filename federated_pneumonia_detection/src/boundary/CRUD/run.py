from typing import Optional, List, Dict, Any
import logging
from sqlalchemy.orm import Session
from federated_pneumonia_detection.src.boundary.CRUD.base import BaseCRUD
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.engine import Run


class RunCRUD(BaseCRUD[Run]):
    """CRUD operations for Run model."""

    def __init__(self):
        super().__init__(Run)
        self.logger = logging.getLogger(__name__)

    def get_by_experiment(self, db: Session, experiment_id: int) -> List[Run]:
        """Get all runs for a specific experiment."""
        return (
            db.query(self.model).filter(self.model.experiment_id == experiment_id).all()
        )

    def get_by_status(self, db: Session, status: str) -> List[Run]:
        """Get runs by status."""
        return db.query(self.model).filter(self.model.status == status).all()

    def get_by_training_mode(self, db: Session, training_mode: str) -> List[Run]:
        """Get runs by training mode."""
        return (
            db.query(self.model).filter(self.model.training_mode == training_mode).all()
        )

    def get_with_config(self, db: Session, id: int) -> Optional[Run]:
        """Get run with its configuration."""
        return db.query(self.model).filter(self.model.id == id).first()

    def get_with_metrics(self, db: Session, id: int) -> Optional[Run]:
        """Get run with all its metrics."""
        return db.query(self.model).filter(self.model.id == id).first()

    def get_with_artifacts(self, db: Session, id: int) -> Optional[Run]:
        """Get run with all its artifacts."""
        return db.query(self.model).filter(self.model.id == id).first()

    def update_status(self, db: Session, id: int, status: str) -> Optional[Run]:
        """Update run status."""
        return self.update(db, id, status=status)

    def get_by_wandb_id(self, db: Session, wandb_id: str) -> Optional[Run]:
        """Get run by W&B ID."""
        return db.query(self.model).filter(self.model.wandb_id == wandb_id).first()

    def get_completed_runs(
        self, db: Session, experiment_id: Optional[int] = None
    ) -> List[Run]:
        """Get all completed runs, optionally filtered by experiment."""
        query = db.query(self.model).filter(self.model.status == "completed")
        if experiment_id:
            query = query.filter(self.model.experiment_id == experiment_id)
        return query.order_by(self.model.end_time.desc()).all()

    def get_failed_runs(
        self, db: Session, experiment_id: Optional[int] = None
    ) -> List[Run]:
        """Get all failed runs, optionally filtered by experiment."""
        query = db.query(self.model).filter(self.model.status == "failed")
        if experiment_id:
            query = query.filter(self.model.experiment_id == experiment_id)
        return query.order_by(self.model.end_time.desc()).all()

    def persist_metrics(
        self, db: Session, run_id: int, epoch_metrics: List[Dict[str, Any]],
        federated_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Persist collected metrics to database.

        Args:
            db: Database session
            run_id: Run ID to associate metrics with
            epoch_metrics: List of epoch metric dictionaries
            federated_context: Optional dict with 'client_id' and 'round_number' for federated mode
        """
        try:
            metrics_to_persist = []
            
            # Extract federated context if provided
            client_id = None
            round_id = None
            if federated_context:
                client_id = federated_context.get('client_id')
                round_number = federated_context.get('round_number', 0)
                
                # If in federated mode, create/get round for this client
                if client_id is not None:
                    from federated_pneumonia_detection.src.boundary.CRUD.round import round_crud
                    round_id = round_crud.get_or_create_round(
                        db, client_id, round_number
                    )
                    self.logger.info(
                        f"[persist_metrics] Federated context: "
                        f"client_id={client_id}, round_id={round_id}"
                    )

            for epoch_data in epoch_metrics:
                epoch = epoch_data.get('epoch', 0)

                # Extract and persist each metric type
                for key, value in epoch_data.items():
                    if key in ['epoch', 'timestamp', 'global_step']:
                        continue

                    if not isinstance(value, (int, float)):
                        continue

                    # Determine dataset type from metric name
                    if key.startswith('train_'):
                        dataset_type = 'train'
                        metric_name = key
                    elif key.startswith('val_'):
                        dataset_type = 'validation'
                        metric_name = key
                    elif key.startswith('test_'):
                        dataset_type = 'test'
                        metric_name = key
                    else:
                        dataset_type = 'other'
                        metric_name = key

                    metric_dict = {
                        'run_id': run_id,
                        'metric_name': metric_name,
                        'metric_value': float(value),
                        'step': epoch,
                        'dataset_type': dataset_type
                    }
                    
                    # Add federated context if applicable
                    if client_id is not None:
                        metric_dict['client_id'] = client_id
                    if round_id is not None:
                        metric_dict['round_id'] = round_id

                    metrics_to_persist.append(metric_dict)

            # Bulk create metrics for efficiency
            if metrics_to_persist:
                run_metric_crud.bulk_create(db, metrics_to_persist)
                db.commit()
                self.logger.info(
                    f"Persisted {len(metrics_to_persist)} metrics to database "
                    f"for run_id={run_id}" +
                    (f", client_id={client_id}, round_id={round_id}" if client_id else "")
                )

        except Exception as e:
            self.logger.error(f"Failed to persist metrics to database: {e}")
            db.rollback()
            raise


run_crud = RunCRUD()
