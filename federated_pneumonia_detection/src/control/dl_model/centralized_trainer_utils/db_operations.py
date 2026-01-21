"""Database operations for training run management."""

import logging
from datetime import datetime
from typing import Optional

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.engine import get_session


def create_training_run(
    source_path: str,
    experiment_name: str,
    logger: logging.Logger,
) -> Optional[int]:
    """
    Create a new training run in the database.

    Args:
        source_path: Path to training data source
        experiment_name: Name of the experiment
        logger: Logger instance

    Returns:
        Run ID if successful, None otherwise
    """
    logger.info("Creating centralized training run in database...")
    db = get_session()
    try:
        run_data = {
            "training_mode": "centralized",
            "status": "in_progress",
            "start_time": datetime.now(),
            "wandb_id": f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "source_path": source_path,
        }
        new_run = run_crud.create(db, **run_data)
        db.commit()
        run_id = new_run.id
        logger.info(f"Created run in database with id={run_id}")
        return run_id
    except Exception as e:
        logger.error(f"Failed to create run: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def complete_training_run(run_id: int, logger: logging.Logger) -> None:
    """
    Mark a training run as completed in the database.

    Args:
        run_id: Run ID to complete
        logger: Logger instance
    """
    logger.info("Marking run as completed in database...")
    db = get_session()
    try:
        run_crud.complete_run(db, run_id=run_id, status="completed")
        db.commit()
        logger.info(f"Run {run_id} marked as completed")
    except Exception as e:
        logger.error(f"Failed to complete run: {e}")
        db.rollback()
    finally:
        db.close()


def fail_training_run(run_id: int, logger: logging.Logger) -> None:
    """
    Mark a training run as failed in the database.

    Args:
        run_id: Run ID to mark as failed
        logger: Logger instance
    """
    db = get_session()
    try:
        run_crud.complete_run(db, run_id=run_id, status="failed")
        db.commit()
        logger.info(f"Run {run_id} marked as failed")
    except Exception as db_error:
        logger.error(f"Failed to mark run as failed: {db_error}")
        db.rollback()
    finally:
        db.close()
