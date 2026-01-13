"""
Database engine configuration with connection pooling and session management.

This module provides a global pooled SQLAlchemy engine and session factory for
database operations. Connection pooling improves performance by reusing database
connections and managing connection lifecycle efficiently.

Key Features:
    - Global engine instance created once at module import time
    - Connection pooling with QueuePool for optimal performance
    - Pre-ping to detect and recover from stale connections
    - Automatic connection recycling to prevent connection exhaustion
    - Context manager support for session lifecycle management

Models are defined in the models/ subdirectory.
"""
import logging
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from federated_pneumonia_detection.config.settings import Settings

from .models import Base

# Initialize settings and logger
settings = Settings()
logger = logging.getLogger(__name__)

# Global engine instance - created once at module import
_engine = None

# Global session factory - bound to the global engine
SessionLocal = None


def _get_engine():
    """
    Create and return a global SQLAlchemy engine with connection pooling.

    This function implements the singleton pattern - the engine is created only
    once at module import time and reused for all subsequent database operations.

    Connection Pool Configuration:
        - poolclass=QueuePool: Standard connection pool with FIFO ordering
        - pool_size=5: Number of persistent connections to maintain
        - max_overflow=10: Maximum additional connections beyond pool_size
        - pool_pre_ping=True: Test connection validity before checkout
        - pool_recycle=3600: Recycle connections after 1 hour (seconds)
        - echo=False: Disable SQL query logging (set True for debugging)
        - connect_args={"timeout": 10}: Connection timeout in seconds

    Returns:
        sqlalchemy.engine.Engine: The global SQLAlchemy engine instance

    Raises:
        SQLAlchemyError: If engine creation `fails
    """
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
            connect_args={"connect_timeout": 10}
        )

        # Create session factory bound to the global engine
        # SessionLocal is a factory function that creates new session instances
        SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)

        logger.info(
            f"Database connection pool initialized: pool_size={5}, "
            f"max_overflow={10}, pool_recycle={3600}s"
        )

        return _engine

    except SQLAlchemyError as e:
        logger.error(f"Failed to create database engine: {e}")
        raise


def create_tables():
    """
    Create all database tables defined by SQLAlchemy models and verify their existence.

    This function uses the global engine to create tables if they don't exist.
    It performs verification checks to ensure all expected tables were created
    successfully and logs the results.

    Returns:
        sqlalchemy.engine.Engine: The global SQLAlchemy engine instance

    Example:
        >>> from federated_pneumonia_detection.src.boundary.engine import create_tables
        >>> engine = create_tables()
        >>> # Output: Creating database tables...
        >>> #         Table verified: runs
        >>> #         Table verified: clients
        >>> #         [OK] All expected tables are present in the database.
    """
    engine = get_engine()
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
    Create a new database session with proper error handling.

    This function returns a new session instance from the SessionLocal factory.
    The session borrows a connection from the connection pool for use during
    database operations.

    Connection Lifecycle:
        1. Session created: Borrows connection from pool
        2. Database operations: Use connection for queries/transactions
        3. session.commit(): Persist changes (if applicable)
        4. session.close(): Return connection to pool (important!)

    Returns:
        sqlalchemy.orm.Session: A new database session instance

    Raises:
        SQLAlchemyError: If session creation fails
    """
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
    Get the global SQLAlchemy engine instance.

    This function returns the global engine, initializing it on first call
    if it hasn't been created yet. The engine is created only once and reused
    for all subsequent database operations.

    Returns:
        sqlalchemy.engine.Engine: The global SQLAlchemy engine instance

    Raises:
        SQLAlchemyError: If engine initialization fails
    """
    global _engine

    # Initialize engine if not already created
    if _engine is None:
        _get_engine()

    return _engine


def dispose_engine():
    """
    Dispose of the global SQLAlchemy engine and close all connections.

    This function should be called during application shutdown to properly
    close all database connections and release resources. It disposes the
    connection pool, closing all idle and active connections.

    IMPORTANT: Call this function before application exit to ensure
    clean shutdown of database connections.

    Raises:
        SQLAlchemyError: If engine disposal fails

    """
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
    """
    Main execution block for testing database setup.

    This block is executed when the script is run directly:
    - Creates database tables
    - Verifies table existence
    - Disposes engine on completion

    Example:
        >>> python -m federated_pneumonia_detection.src.boundary.engine
    """
    try:
        create_tables()
        logger.info("Database setup completed successfully")
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise
    finally:
        # Clean up connections
        dispose_engine()
