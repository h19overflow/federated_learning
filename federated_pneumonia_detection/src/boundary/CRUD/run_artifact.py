from typing import Optional, List
from sqlalchemy.orm import Session
from federated_pneumonia_detection.src.boundary.CRUD.base import BaseCRUD
from federated_pneumonia_detection.src.boundary.engine import RunArtifact


class RunArtifactCRUD(BaseCRUD[RunArtifact]):
    """CRUD operations for RunArtifact model."""

    def __init__(self):
        super().__init__(RunArtifact)

    def get_by_run(self, db: Session, run_id: int) -> List[RunArtifact]:
        """Get all artifacts for a specific run."""
        return db.query(self.model).filter(self.model.run_id == run_id).all()

    def get_by_type(
        self, db: Session, run_id: int, artifact_type: str
    ) -> List[RunArtifact]:
        """Get artifacts by type for a specific run."""
        return (
            db.query(self.model)
            .filter(
                self.model.run_id == run_id, self.model.artifact_type == artifact_type
            )
            .all()
        )

    def get_by_name(
        self, db: Session, run_id: int, artifact_name: str
    ) -> Optional[RunArtifact]:
        """Get a specific artifact by name."""
        return (
            db.query(self.model)
            .filter(
                self.model.run_id == run_id, self.model.artifact_name == artifact_name
            )
            .first()
        )

    def get_models(self, db: Session, run_id: int) -> List[RunArtifact]:
        """Get all model artifacts for a run."""
        return self.get_by_type(db, run_id, "model")

    def get_images(self, db: Session, run_id: int) -> List[RunArtifact]:
        """Get all image artifacts for a run."""
        return self.get_by_type(db, run_id, "image")

    def get_logs(self, db: Session, run_id: int) -> List[RunArtifact]:
        """Get all log artifacts for a run."""
        return self.get_by_type(db, run_id, "log")

    def search_by_name(
        self, db: Session, run_id: int, search_term: str
    ) -> List[RunArtifact]:
        """Search artifacts by name pattern."""
        return (
            db.query(self.model)
            .filter(
                self.model.run_id == run_id,
                self.model.artifact_name.ilike(f"%{search_term}%"),
            )
            .all()
        )


run_artifact_crud = RunArtifactCRUD()
