from typing import Optional, List
from sqlalchemy.orm import Session
from federated_pneumonia_detection.src.boundary.CRUD.base import BaseCRUD
from federated_pneumonia_detection.src.boundary.engine import Experiment


class ExperimentCRUD(BaseCRUD[Experiment]):
    """CRUD operations for Experiment model."""
    
    def __init__(self):
        super().__init__(Experiment)
    
    def get_by_name(self, db: Session, name: str) -> Optional[Experiment]:
        """Get experiment by name."""
        return db.query(self.model).filter(self.model.name == name).first()
    
    def get_with_runs(self, db: Session, id: int) -> Optional[Experiment]:
        """Get experiment with all its runs."""
        return db.query(self.model).filter(self.model.id == id).first()
    
    def get_recent(self, db: Session, limit: int = 10) -> List[Experiment]:
        """Get most recent experiments."""
        return db.query(self.model).order_by(self.model.created_at.desc()).limit(limit).all()
    
    def search_by_name(self, db: Session, search_term: str) -> List[Experiment]:
        """Search experiments by name pattern."""
        return db.query(self.model).filter(
            self.model.name.ilike(f"%{search_term}%")
        ).all()


experiment_crud = ExperimentCRUD()
