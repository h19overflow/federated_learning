from contextlib import contextmanager
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.engine import Base, get_session

ModelType = TypeVar("ModelType", bound=Base)


class BaseCRUD(Generic[ModelType]):
    """Base CRUD class with generic operations for database models."""

    def __init__(self, model: Type[ModelType]):
        self.model = model

    @contextmanager
    def get_db_session(self):
        """Context manager for database sessions with automatic cleanup."""
        session = get_session()
        try:
            yield session
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def create(self, db: Session, **kwargs) -> ModelType:
        """Create a new record."""
        db_obj = self.model(**kwargs)
        db.add(db_obj)
        db.flush()
        db.refresh(db_obj)
        return db_obj

    def get(self, db: Session, id: int) -> Optional[ModelType]:
        """Get a single record by ID."""
        return db.query(self.model).filter(self.model.id == id).first()

    def get_multi(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ModelType]:
        """Get multiple records with optional filtering."""
        query = db.query(self.model)

        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.filter(getattr(self.model, key) == value)

        return query.offset(skip).limit(limit).all()

    def update(self, db: Session, id: int, **kwargs) -> Optional[ModelType]:
        """Update a record by ID."""
        db_obj = self.get(db, id)
        if db_obj:
            for key, value in kwargs.items():
                if hasattr(db_obj, key):
                    setattr(db_obj, key, value)
            db.flush()
            db.refresh(db_obj)
        return db_obj

    def delete(self, db: Session, id: int) -> bool:
        """Delete a record by ID."""
        db_obj = self.get(db, id)
        if db_obj:
            db.delete(db_obj)
            db.flush()
            return True
        return False

    def count(self, db: Session, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records with optional filtering."""
        query = db.query(self.model)

        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.filter(getattr(self.model, key) == value)

        return query.count()

    def exists(self, db: Session, id: int) -> bool:
        """Check if a record exists by ID."""
        return db.query(self.model).filter(self.model.id == id).first() is not None

    def bulk_create(
        self,
        db: Session,
        objects: List[Dict[str, Any]],
    ) -> List[ModelType]:
        """Bulk create records for efficiency."""
        db_objs = [self.model(**obj) for obj in objects]
        db.add_all(db_objs)
        db.flush()
        return db_objs
