"""
Database engine configuration and session management.

This module provides database connection utilities and session factories.
Models are defined in the models/ subdirectory.
"""
import logging
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from federated_pneumonia_detection.config.settings import Settings

from .models import Base, Run, Client, Round, RunMetric, ServerEvaluation

settings = Settings()
logger = logging.getLogger(__name__)


def create_tables():
    """
    Create all database tables defined by SQLAlchemy models and verify their existence.

    Returns:
        SQLAlchemy Engine instance
    """
    engine = create_engine(settings.get_postgres_db_uri())
    logger.info("Creating database tables...")
    Base.metadata.create_all(engine)
    
    # Verification check
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    expected_tables = Base.metadata.tables.keys()
    
    logger.info(f"Existing tables in database: {existing_tables}")
    
    created_all = True
    for table in expected_tables:
        if table in existing_tables:
            logger.info(f"Table verified: {table}")
        else:
            logger.error(f"Table MISSING: {table}")
            created_all = False
            
    if created_all:
        logger.info("[OK] All expected tables are present in the database.")
    else:
        logger.warning("[!] Some expected tables were not found after creation attempt.")
        
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
