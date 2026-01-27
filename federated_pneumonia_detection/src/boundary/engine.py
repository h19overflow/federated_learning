"""
Database engine and session management with connection pooling.

Provides a singleton SQLAlchemy engine with QueuePool, pre-ping for stale connections,
and automatic recycling. Migration-aware: use create_tables() for dev or Alembic for production.  # noqa: E501
"""

import logging
import os

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from federated_pneumonia_detection.config.settings import get_settings

from .models import Base

# Initialize settings and logger
settings = get_settings()
logger = logging.getLogger(__name__)

# Global engine instance - created once at module import
_engine = None

# Global session factory - bound to the global engine
SessionLocal = None


def _get_engine():
    """Create global SQLAlchemy engine with QueuePool (5 connections, 10 overflow, 1hr recycle)."""  # noqa: E501
    global _engine, SessionLocal

    if _engine is not None:
        # Engine already initialized, return existing instance
        return _engine

    try:
        # Create engine with connection pooling configuration
        _engine = create_engine(
            settings.get_postgres_db_uri(),
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
            connect_args={"connect_timeout": 10},
        )

        # Create session factory bound to the global engine
        # SessionLocal is a factory function that creates new session instances
        SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)

        logger.info(
            f"Database connection pool initialized: pool_size={5}, "
            f"max_overflow={10}, pool_recycle={3600}s",
        )

        return _engine

    except SQLAlchemyError as e:
        logger.error(f"Failed to create database engine: {e}")
        raise


def create_tables(force: bool = False):
    """Create and verify all database tables. Respects USE_ALEMBIC env var (production mode)."""  # noqa: E501
    engine = get_engine()
    use_alembic = os.getenv("USE_ALEMBIC", "false").lower() == "true"

    # Check if Alembic should be used instead
    if use_alembic and not force:
        logger.warning(
            "[MIGRATION-AWARE] USE_ALEMBIC=true detected. "
            "Schema changes should be managed via Alembic migrations.",
        )
        logger.warning("To apply migrations, run: alembic upgrade head")
        logger.warning("To force table creation anyway, call create_tables(force=True)")
        logger.info(
            "Skipping Base.metadata.create_all() - using Alembic for schema management",
        )

        # Verify tables exist (but don't create them)
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        expected_tables = Base.metadata.tables.keys()

        all_exist = True
        for table in expected_tables:
            if table in existing_tables:
                logger.info(f"Table verified: {table}")
            else:
                logger.error(f"Table MISSING: {table} - run 'alembic upgrade head'")
                all_exist = False

        if not all_exist:
            raise ValueError(
                "Database schema is incomplete. Run 'alembic upgrade head' to apply migrations.",  # noqa: E501
            )

        return engine

    # Development mode or force=True: Create tables directly
    if force:
        logger.info(
            "[FORCE MODE] Creating database tables via Base.metadata.create_all()...",
        )
    else:
        logger.info("Creating database tables via Base.metadata.create_all()...")

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
        logger.warning(
            "[!] Some expected tables were not found after creation attempt.",
        )

    return engine


def get_session():
    """Create a new database session (borrows connection from pool, call .close() to return)."""  # noqa: E501
    try:
        # Initialize engine if not already created
        if SessionLocal is None:
            get_engine()

        # Return new session instance
        return SessionLocal()

    except SQLAlchemyError as e:
        logger.error(f"Failed to create database session: {e}")
        raise


def get_engine():
    """
    Get or initialize the global SQLAlchemy engine instance (singleton).

    Returns:
        sqlalchemy.engine.Engine: The global SQLAlchemy engine with connection pooling
    """
    global _engine

    # Initialize engine if not already created
    if _engine is None:
        _get_engine()

    return _engine


def dispose_engine():
    """Dispose engine and close all connections (call on app shutdown)."""
    global _engine, SessionLocal

    try:
        if _engine is not None:
            logger.info("Disposing database connection pool...")
            _engine.dispose()
            _engine = None
            SessionLocal = None
            logger.info("Database connection pool disposed successfully")
        else:
            logger.warning("No database engine to dispose (engine is None)")

    except SQLAlchemyError as e:
        logger.error(f"Failed to dispose database engine: {e}")
        raise


# Initialize engine at module import time
_get_engine()


if __name__ == "__main__":
    """Create tables and dispose engine on exit."""
    try:
        create_tables()
        logger.info("Database setup completed successfully")
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise
    finally:
        # Clean up connections
        dispose_engine()
