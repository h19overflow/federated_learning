"""
Endpoints for running centralized training experiments.

This module provides HTTP endpoints to trigger centralized machine learning training
on the pneumonia dataset. The training is executed asynchronously in the background,
allowing the endpoint to return immediately while training proceeds.

The training process:
1. Loads the dataset from the configured Training directory
2. Initializes a CentralizedTrainer with current configuration settings
3. Trains the model using standard supervised learning approach
4. Stores results and checkpoints in configured output directories

Configuration should be set prior to invoking training via the configuration endpoints.
"""

from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form
from typing import Dict, Any
import os
import shutil

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import (
    CentralizedTrainer,
)

router = APIRouter(
    prefix="/experiments/centralized",
    tags=["experiments", "centralized"],
)

logger = get_logger(__name__)


def _run_centralized_training_task(
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
        task_logger.info(f"\nInitializing CentralizedTrainer...")
        task_logger.info(f"  Source: {source_path}")
        task_logger.info(f"  Checkpoints: {checkpoint_dir}")
        task_logger.info(f"  Logs: {logs_dir}")
        task_logger.info(f"  Experiment: {experiment_name}")

        trainer = CentralizedTrainer(
            config_path=None, checkpoint_dir=checkpoint_dir, logs_dir=logs_dir
        )

        status = trainer.get_training_status()
        task_logger.info("\nTrainer Configuration:")
        task_logger.info(f"  Epochs: {status['config']['epochs']}")
        task_logger.info(f"  Learning Rate: {status['config']['learning_rate']}")
        task_logger.info(f"  Batch Size: {status['config']['batch_size']}")
        task_logger.info(f"  Validation Split: {status['config']['validation_split']}")

        task_logger.info("\nStarting training...")
        task_logger.info("-" * 80)

        results = trainer.train(
            source_path=source_path,
            experiment_name=experiment_name,
            csv_filename=csv_filename,
        )

        task_logger.info("\n" + "=" * 80)
        task_logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        task_logger.info("=" * 80)
        task_logger.info("\nResults Summary:")
        task_logger.info(f"  Status: {results.get('status', 'completed')}")
        task_logger.info(f"  Best model: {results.get('best_checkpoint_path', 'N/A')}")
        task_logger.info(f"  Checkpoint directory: {checkpoint_dir}")
        task_logger.info(f"  Logs directory: {logs_dir}")

        if "final_metrics" in results:
            task_logger.info("\nFinal Metrics:")
            for key, value in results["final_metrics"].items():
                task_logger.info(f"  {key}: {value}")

        return results

    except Exception as e:
        task_logger.error("\n" + "=" * 80)
        task_logger.error("TRAINING FAILED!")
        task_logger.error("=" * 80)
        task_logger.error(f"Error: {type(e).__name__}: {str(e)}")

        import traceback

        task_logger.error("\nFull traceback:")
        task_logger.error(traceback.format_exc())

        return {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__,
        }


@router.post("/train")
async def start_centralized_training(
    background_tasks: BackgroundTasks,
    data_zip: UploadFile = File(...),
    checkpoint_dir: str = Form("results/centralized/checkpoints"),
    logs_dir: str = Form("results/centralized/logs"),
    experiment_name: str = Form("pneumonia_centralized"),
    csv_filename: str = Form("stage2_train_metadata.csv"),
) -> Dict[str, Any]:
    """
    Start centralized training in the background with uploaded data.

    Initiates a centralized machine learning training process using the current
    configuration settings. The training runs asynchronously, allowing this endpoint
    to return immediately.

    **Training Process:**
    - Extracts uploaded data archive (Images/ and metadata CSV)
    - Uses all training samples in a single centralized trainer
    - Applies data augmentation and preprocessing based on configuration
    - Trains using standard supervised learning approach
    - Performs validation during training with early stopping
    - Saves best model checkpoints and training logs

    **Prerequisites:**
    - Configuration should be set via `/configuration/set_configuration` endpoint
    - Upload a ZIP file containing Images/ directory and metadata CSV

    **Parameters:**
    - `data_zip`: ZIP file containing Images/ directory and metadata CSV (required)
    - `checkpoint_dir`: Where to save model checkpoints (default: "results/centralized/checkpoints")
    - `logs_dir`: Where to save training logs (default: "results/centralized/logs")
    - `experiment_name`: Identifier for this training run (default: "pneumonia_centralized")
    - `csv_filename`: Metadata CSV filename inside archive (default: "stage2_train_metadata.csv")

    **Response:**
    Returns immediately with confirmation that training has been queued. Check logs
    in the specified `logs_dir` for training progress and results.

    **Status Tracking:**
    Monitor training progress through:
    - Log files at `{logs_dir}/`
    - Checkpoint files at `{checkpoint_dir}/`
    """
    import zipfile
    import tempfile

    temp_dir = None
    try:
        # Create temp directory for extraction
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, data_zip.filename)

        # Save uploaded file
        with open(zip_path, "wb") as f:
            content = await data_zip.read()
            f.write(content)

        # Extract archive
        extract_path = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        source_path = extract_path

        logger.info(
            f"Received request to start centralized training: {experiment_name}"
        )
        logger.info(f"Extracted data to: {source_path}")

        background_tasks.add_task(
            _run_centralized_training_task,
            source_path=source_path,
            checkpoint_dir=checkpoint_dir,
            logs_dir=logs_dir,
            experiment_name=experiment_name,
            csv_filename=csv_filename,
        )

        return {
            "message": "Centralized training started successfully",
            "experiment_name": experiment_name,
            "checkpoint_dir": checkpoint_dir,
            "logs_dir": logs_dir,
            "status": "queued",
        }
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
