"""
Helper functions for federated learning training tasks.

Provides modular functions for path validation, configuration loading,
data loading, training execution, and error handling.
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.src.control.federated_learning import (
    FederatedTrainer,
)
from federated_pneumonia_detection.src.utils.data_processing import load_metadata


def validate_training_paths(source_path: str, csv_filename: str) -> tuple[Path, Path, Path]:
    """
    Validate that all required training paths exist.

    Args:
        source_path: Path to training data directory
        csv_filename: Name of the CSV metadata file

    Returns:
        Tuple of (source_path, image_dir, metadata_path) as Path objects

    Raises:
        FileNotFoundError: If any required path doesn't exist
    """
    print("Validating training paths...")
    source_path_obj = Path(source_path)
    image_dir = source_path_obj / "Images"
    metadata_path = source_path_obj / csv_filename

    if not source_path_obj.exists():
        raise FileNotFoundError(f"Source path not found: {source_path_obj}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    return source_path_obj, image_dir, metadata_path


def load_training_config(task_logger) -> tuple[ExperimentConfig, SystemConstants, torch.device]:
    """
    Load and log training configuration.

    Args:
        task_logger: Logger instance for logging info

    Returns:
        Tuple of (config, constants, device)
    """
    task_logger.info("\nLoading configuration...")
    constants = SystemConstants()
    config = ExperimentConfig()

    task_logger.info(f"  Num clients: {config.num_clients}")
    task_logger.info(f"  Num rounds: {config.num_rounds}")
    task_logger.info(f"  Local epochs: {config.local_epochs}")
    task_logger.info(f"  Learning rate: {config.learning_rate}")
    task_logger.info(f"  Batch size: {config.batch_size}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        task_logger.info("  Device: GPU (CUDA)")
    else:
        device = torch.device("cpu")
        task_logger.info("  Device: CPU")

    return config, constants, device


def load_training_data(
    metadata_path: Path, constants: SystemConstants, task_logger
) -> pd.DataFrame:
    """
    Load and log training dataset metadata.

    Args:
        metadata_path: Path to metadata CSV file
        constants: System constants with column names
        task_logger: Logger instance

    Returns:
        DataFrame with training data metadata
    """
    task_logger.info("\nLoading dataset metadata...")
    data_df = load_metadata(metadata_path, constants, task_logger)
    task_logger.info(f"  Total samples: {len(data_df)}")
    class_dist = data_df[constants.TARGET_COLUMN].value_counts().to_dict()
    task_logger.info(f"  Class distribution: {class_dist}")
    return data_df


def execute_federated_training(
    trainer: FederatedTrainer,
    data_df: pd.DataFrame,
    image_dir: Path,
    experiment_name: str,
    task_logger,
) -> Dict[str, Any]:
    """
    Execute the federated learning training pipeline.

    Args:
        trainer: FederatedTrainer instance
        data_df: Training data metadata DataFrame
        image_dir: Path to images directory
        experiment_name: Name of the experiment
        task_logger: Logger instance

    Returns:
        Dictionary containing training results
    """
    task_logger.info("\nStarting federated learning simulation...")
    task_logger.info("-" * 80)

    results = trainer.train(
        data_df=data_df,
        image_dir=image_dir,
        experiment_name=experiment_name,
    )

    task_logger.info("\n" + "=" * 80)
    task_logger.info("FEDERATED TRAINING COMPLETED SUCCESSFULLY!")
    task_logger.info("=" * 80)

    return results


def handle_training_error(task_logger, e: Exception) -> Dict[str, Any]:
    """
    Handle federated training errors and return error response.

    Args:
        task_logger: Logger instance
        e: Exception that occurred

    Returns:
        Dictionary with error status and details
    """
    task_logger.error("FEDERATED TRAINING FAILED!")
    task_logger.error(f"Error: {type(e).__name__}: {str(e)}")

    import traceback

    task_logger.error("\nFull traceback:")
    task_logger.error(traceback.format_exc())

    return {
        "status": "failed",
        "error": str(e),
        "error_type": type(e).__name__,
    }
