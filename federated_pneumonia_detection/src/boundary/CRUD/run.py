from typing import Optional, List
from sqlalchemy.orm import Session
from federated_pneumonia_detection.src.boundary.CRUD.base import BaseCRUD
from federated_pneumonia_detection.src.boundary.engine import Run


class RunCRUD(BaseCRUD[Run]):
    """CRUD operations for Run model."""
    
    def __init__(self):
        super().__init__(Run)
    
    def get_by_experiment(self, db: Session, experiment_id: int) -> List[Run]:
        """Get all runs for a specific experiment."""
        return db.query(self.model).filter(self.model.experiment_id == experiment_id).all()
    
    def get_by_status(self, db: Session, status: str) -> List[Run]:
        """Get runs by status."""
        return db.query(self.model).filter(self.model.status == status).all()
    
    def get_by_training_mode(self, db: Session, training_mode: str) -> List[Run]:
        """Get runs by training mode."""
        return db.query(self.model).filter(self.model.training_mode == training_mode).all()
    
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
    
    def get_completed_runs(self, db: Session, experiment_id: Optional[int] = None) -> List[Run]:
        """Get all completed runs, optionally filtered by experiment."""
        query = db.query(self.model).filter(self.model.status == 'completed')
        if experiment_id:
            query = query.filter(self.model.experiment_id == experiment_id)
        return query.order_by(self.model.end_time.desc()).all()
    
    def get_failed_runs(self, db: Session, experiment_id: Optional[int] = None) -> List[Run]:
        """Get all failed runs, optionally filtered by experiment."""
        query = db.query(self.model).filter(self.model.status == 'failed')
        if experiment_id:
            query = query.filter(self.model.experiment_id == experiment_id)
        return query.order_by(self.model.end_time.desc()).all()


run_crud = RunCRUD()
