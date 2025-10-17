from typing import Optional, List
from sqlalchemy.orm import Session
from federated_pneumonia_detection.src.boundary.CRUD.base import BaseCRUD
from federated_pneumonia_detection.src.boundary.engine import RunConfiguration


class RunConfigurationCRUD(BaseCRUD[RunConfiguration]):
    """CRUD operations for RunConfiguration model."""
    
    def __init__(self):
        super().__init__(RunConfiguration)
    
    def get_by_run(self, db: Session, run_id: int) -> Optional[RunConfiguration]:
        """Get configuration for a specific run."""
        return db.query(self.model).filter(self.model.run_id == run_id).first()
    
    def get_by_partition_strategy(self, db: Session, strategy: str) -> List[RunConfiguration]:
        """Get configurations by partition strategy."""
        return db.query(self.model).filter(self.model.partition_strategy == strategy).all()
    
    def get_by_learning_rate(self, db: Session, learning_rate: float) -> List[RunConfiguration]:
        """Get configurations by learning rate."""
        return db.query(self.model).filter(self.model.learning_rate == learning_rate).all()
    
    def get_by_hyperparameters(
        self,
        db: Session,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None
    ) -> List[RunConfiguration]:
        """Get configurations matching specific hyperparameters."""
        query = db.query(self.model)
        
        if learning_rate is not None:
            query = query.filter(self.model.learning_rate == learning_rate)
        if batch_size is not None:
            query = query.filter(self.model.batch_size == batch_size)
        if epochs is not None:
            query = query.filter(self.model.epochs == epochs)
        
        return query.all()


run_configuration_crud = RunConfigurationCRUD()
