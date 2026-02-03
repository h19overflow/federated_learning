import logging

from federated_pneumonia_detection.src.boundary.engine import create_tables

logger = logging.getLogger(__name__)

def initialize_database() -> None:
    """Create database tables. Raises on failure (critical service)."""
    try:
        logger.info("Ensuring database tables exist...")
        create_tables()
        logger.info("Database tables verified/created")
    except Exception as e:
        logger.critical(f"DATABASE INITIALIZATION FAILED: {e}")
        logger.critical("Cannot proceed with startup. Shutting down.")
        raise

def shutdown_database() -> None:
    """Dispose database connections."""
    try:
        from federated_pneumonia_detection.src.boundary.engine import dispose_engine

        dispose_engine()
        logger.info("Database connections disposed")
    except Exception as e:
        logger.warning(f"Error disposing database connections: {e}")
