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
    task_logger = get_logger(f"{__name__}._task")

    task_logger.info("=" * 80)
    task_logger.info("CENTRALIZED TRAINING - Pneumonia Detection (Background Task)")
    task_logger.info("=" * 80)

    try:
        task_logger.info(f"  Source: {source_path}")
        config_path = r"federated_pneumonia_detection\config\default_config.yaml"
        trainer = CentralizedTrainer(
            config_path=config_path,
            checkpoint_dir=checkpoint_dir,
            logs_dir=logs_dir,
        )

        task_logger.info("\nTrainer Configuration:")

        results = trainer.train(
            source_path=source_path,
            experiment_name=experiment_name,
            csv_filename=csv_filename,
        )

        task_logger.info("\n" + "=" * 80)
        task_logger.info("TRAINING COMPLETED SUCCESSFULLY!")

        if "final_metrics" in results:
            task_logger.info("\nFinal Metrics:")
            for key, value in results["final_metrics"].items():
                task_logger.info(f"  {key}: {value}")

        return results
    except Exception as e:
        task_logger.error(f"Error: {type(e).__name__}: {str(e)}")
        return {"status": "failed", "error": str(e)}
