"""
Centralized training task functions.

Provides background task execution for centralized machine learning training.
"""

from typing import Any, Dict

from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import (
    CentralizedTrainer,
)
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger


def run_centralized_training_task(
    source_path: str,
    checkpoint_dir: str,
    logs_dir: str,
    experiment_name: str,
    csv_filename: str,
) -> Dict[str, Any]:
    """
    Background task to execute centralized training.

    Args:
        source_path: Path to training data directory
        checkpoint_dir: Directory to save model checkpoints
        logs_dir: Directory to save training logs
        experiment_name: Name identifier for this training run
        csv_filename: Name of the CSV metadata file

    Returns:
        Dictionary containing training results
    """
    task_logger = get_logger("centralized_training_task")
    task_logger.info(f"[BACKGROUND TASK] Starting centralized training: experiment={experiment_name}")
    task_logger.info(f"[BACKGROUND TASK] source_path={source_path}, csv={csv_filename}")
    try:
        config_path = r"federated_pneumonia_detection\config\default_config.yaml"
        trainer = CentralizedTrainer(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            logs_dir=logs_dir,
        )
        results = trainer.train(
            source_path=source_path,
            experiment_name=experiment_name,
            csv_filename=csv_filename,
        )

        return results
    except Exception as e:
        task_logger.error(f"Error: {type(e).__name__}: {str(e)}")
        return {"status": "failed", "error": str(e)}
