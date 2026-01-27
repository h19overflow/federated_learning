"""Database operations for training run management."""

import logging
from datetime import datetime
from typing import Optional, cast

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.control.analytics.internals.services.final_epoch_stats_service import (  # noqa: E501
    FinalEpochStatsService,
)


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
        return cast(Optional[int], run_id)
    except Exception as e:
        logger.error(f"Failed to create run: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def _get_final_epoch_cm(db, run_id: int):
    """
    Get confusion matrix values from final epoch for centralized run.

    Delegates to FinalEpochStatsService.get_cm_centralized().

    Returns:
        Dict with keys: true_positives, true_negatives, false_positives, false_negatives, epoch  # noqa: E501
        or None if incomplete data
    """
    return FinalEpochStatsService.get_cm_centralized(db, run_id)


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

        # Compute and persist final epoch stats using service
        FinalEpochStatsService.calculate_and_persist_centralized(db, run_id)

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
