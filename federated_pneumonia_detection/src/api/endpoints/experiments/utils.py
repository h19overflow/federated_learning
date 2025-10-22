from pathlib import Path
from typing import Dict, Any
from fastapi import UploadFile
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import CentralizedTrainer
import zipfile
import tempfile
import os
import shutil

LOGS_DIR = Path("logs/progress")

def find_experiment_log_file(experiment_id: str) -> Path | None:
    """
    Find the log file for a given experiment ID.

    Args:
        experiment_id: Experiment identifier

    Returns:
        Path to log file or None if not found
    """
    if not LOGS_DIR.exists():
        return None

    # Search for log files matching the experiment ID
    for log_file in LOGS_DIR.glob("*.json"):
        if experiment_id in log_file.stem:
            return log_file

    return None



def calculate_progress(log_data: Dict[str, Any]) -> float:
    """
    Calculate training progress percentage.

    Args:
        log_data: Experiment log data

    Returns:
        Progress percentage (0-100)
    """
    epochs_data = log_data.get("epochs", [])

    # Try to determine total epochs from metadata or config
    # This is a simplified approach - may need adjustment based on actual log structure
    if not epochs_data:
        return 0.0

    # Find epoch_end events to count completed epochs
    completed_epochs = sum(1 for e in epochs_data if e.get("type") == "epoch_end")

    # Try to find total epochs from config or infer from data
    total_epochs = 10  # Default fallback

    # Look for total_epochs in any epoch_start event
    for event in epochs_data:
        if event.get("type") == "epoch_start" and "total_epochs" in event:
            total_epochs = event["total_epochs"]
            break

    if total_epochs > 0:
        return min(100.0, (completed_epochs / total_epochs) * 100)

    return 0.0

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
        websocket_manager: Optional WebSocket manager for real-time progress updates

    Returns:
        Dictionary containing training results
    """
    import asyncio
    task_logger = get_logger(f"{__name__}._task")

    task_logger.info("=" * 80)
    task_logger.info("CENTRALIZED TRAINING - Pneumonia Detection (Background Task)")
    task_logger.info("=" * 80)

    try:
        task_logger.info(f"  Source: {source_path}")

        trainer = CentralizedTrainer(
            config_path=None,
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
                    # Send completion status via WebSocket
        return results
    except Exception as e:
            task_logger.error(f"Error: {type(e).__name__}: {str(e)}")
            return {"status": "failed", "error": str(e)}


async def prepare_zip(data_zip:UploadFile,logger,experiment_name):
    temp_dir =None
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
        return source_path
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise

       