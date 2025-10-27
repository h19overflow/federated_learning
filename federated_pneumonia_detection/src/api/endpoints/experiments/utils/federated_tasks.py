"""
Federated learning training task functions.

Provides background task execution for federated machine learning training.
"""

from typing import Any, Dict

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from federated_pneumonia_detection.src.control.federated_learning import (
    FederatedTrainer,
)
from .federated_helpers import (
    execute_federated_training,
    handle_training_error,
    load_training_config,
    load_training_data,
    validate_training_paths,
)


def run_federated_training_task(
    source_path: str,
    experiment_name: str,
    csv_filename: str,
) -> Dict[str, Any]:
    """
    Background task to execute federated training.

    Args:
        source_path: Path to training data directory
        experiment_name: Name identifier for this training run
        csv_filename: Name of the CSV metadata file

    Returns:
        Dictionary containing training results
    """
    task_logger = get_logger(f"{__name__}._task")

    task_logger.info("=" * 80)
    task_logger.info(
        "FEDERATED LEARNING TRAINING - Pneumonia Detection (Background Task)"
    )
    task_logger.info("=" * 80)

    try:
        source_path_obj, image_dir, metadata_path = validate_training_paths(
            source_path, csv_filename
        )
        config, constants, device = load_training_config(task_logger)
        data_df = load_training_data(metadata_path, constants, task_logger)

        task_logger.info("\nInitializing FederatedTrainer...")
        trainer = FederatedTrainer(
            config=config,
            constants=constants,
            device=device,

        )

        return execute_federated_training(
            trainer, data_df, image_dir, experiment_name, task_logger
        )

    except Exception as e:
        return handle_training_error(task_logger, e)
