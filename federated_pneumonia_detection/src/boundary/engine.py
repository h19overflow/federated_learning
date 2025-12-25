"""
Database engine configuration and session management.

This module provides database connection utilities and session factories.
Models are defined in the models/ subdirectory.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from federated_pneumonia_detection.config.settings import Settings

from .models import Base, Run, Client, Round, RunMetric, ServerEvaluation

settings = Settings()


def create_tables():
    """
    Create all database tables defined by SQLAlchemy models.

    Returns:
        SQLAlchemy Engine instance
    """
    engine = create_engine(settings.get_postgres_db_uri())
    Base.metadata.create_all(engine)
    return engine


def get_session():
    """
    Create a new database session.

    Returns:
        SQLAlchemy Session instance
    """
    engine = create_engine(settings.get_postgres_db_uri())
    Session = sessionmaker(bind=engine)
    return Session()


def get_engine():
    """
    Get the SQLAlchemy engine instance.

    Returns:
        SQLAlchemy Engine instance
    """
    engine = create_engine(settings.get_postgres_db_uri())
    return engine


if __name__ == "__main__":
    create_tables()
