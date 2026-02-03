import logging

from federated_pneumonia_detection.src.control.dl_model.internals.data.wandb_inference_tracker import (  # noqa: E501
    get_wandb_tracker,
)

logger = logging.getLogger(__name__)

def initialize_wandb_tracker() -> None:
    """Initialize W&B inference tracker for experiment tracking."""
    try:
        tracker = get_wandb_tracker()
        if tracker.initialize(
            entity="projectontheside25-multimedia-university",
            project="FYP2",
            job_type="inference",
        ):
            logger.info(
                "W&B inference tracker initialized (experiment tracking enabled)",
            )
        else:
            logger.warning(
                "W&B tracker rejected configuration (check credentials). "
                "Experiment tracking will be unavailable.",
            )
    except ConnectionError as e:
        logger.warning(
            f"W&B connection failed: {e} "
            "(experiment tracking disabled, but training continues)",
        )
    except ImportError as e:
        logger.warning(
            f"W&B not installed: {e} (install with: pip install wandb) "
            "(experiment tracking disabled)",
        )
    except Exception as e:
        logger.warning(
            f"W&B initialization failed (unexpected): {e} "
            "(experiment tracking will be unavailable)",
        )

def shutdown_wandb_tracker() -> None:
    """Finish W&B tracking session."""
    try:
        tracker = get_wandb_tracker()
        tracker.finish()
        logger.info("W&B inference tracker shutdown complete")
    except Exception as e:
        logger.warning(
            f"W&B tracker shutdown had issues: {e} "
            "(this is non-fatal, app still shutting down)",
        )
