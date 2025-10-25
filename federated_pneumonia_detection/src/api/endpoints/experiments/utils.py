from pathlib import Path
from typing import Dict, Any,List
from fastapi import UploadFile
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import CentralizedTrainer
import zipfile
import tempfile
import os
import shutil
from collections import defaultdict

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
        websocket_manager: Optional WebSocket manager for real-time progress updates

    Returns:
        Dictionary containing training results
    """
    task_logger = get_logger(f"{__name__}._task")

    task_logger.info("=" * 80)
    task_logger.info("CENTRALIZED TRAINING - Pneumonia Detection (Background Task)")
    task_logger.info("=" * 80)

    try:
        task_logger.info(f"  Source: {source_path}")
        config_path = r'federated_pneumonia_detection\config\default_config.yaml'
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

def _transform_run_to_results(run) -> Dict[str, Any]:
    """
    Transform database Run object to ExperimentResults format.

    Database format:
        RunMetric(metric_name='val_recall', metric_value=0.95, step=10, dataset_type='validation')

    Frontend format:
        {
          final_metrics: {accuracy: 0.92, ...},
          training_history: [{epoch: 0, train_loss: 0.5, ...}],
          ...
        }
    """

    # Group metrics by epoch (step)
    metrics_by_epoch = defaultdict(dict)
    final_metrics = {}

    for metric in run.metrics:
        epoch = metric.step
        metric_name = metric.metric_name
        value = metric.metric_value

        # Store in epoch-based structure
        metrics_by_epoch[epoch][metric_name] = value

        # Track final (last epoch) metrics
        if epoch >= max(metrics_by_epoch.keys(), default=0):
            if metric_name in ['val_accuracy', 'val_acc', 'val_precision', 'val_recall',
                              'val_f1', 'val_auroc', 'val_auc', 'val_loss']:
                final_metrics[metric_name] = value

    # Build training history
    training_history = []
    for epoch in sorted(metrics_by_epoch.keys()):
        epoch_data = metrics_by_epoch[epoch]

        training_history.append({
            "epoch": epoch + 1,  # Convert from 0-indexed to 1-indexed for display
            "train_loss": epoch_data.get("train_loss", 0.0),
            "val_loss": epoch_data.get("val_loss", 0.0),
            "train_acc": epoch_data.get("train_accuracy", epoch_data.get("train_acc", 0.0)),
            "val_acc": epoch_data.get("val_accuracy", epoch_data.get("val_acc", 0.0)),
            "train_f1": epoch_data.get("train_f1", 0.0),
            "val_precision": epoch_data.get("val_precision", 0.0),
            "val_recall": epoch_data.get("val_recall", 0.0),
            "val_f1": epoch_data.get("val_f1", 0.0),
            "val_auroc": epoch_data.get("val_auroc", epoch_data.get("val_auc", 0.0)),
        })

    # Extract final metrics (use last epoch values or best values)
    last_epoch_data = metrics_by_epoch[max(metrics_by_epoch.keys())] if metrics_by_epoch else {}

    # Calculate final metrics from last epoch
    accuracy = final_metrics.get("val_accuracy", final_metrics.get("val_acc", 0.0))
    precision = final_metrics.get("val_precision", 0.0)
    recall = final_metrics.get("val_recall", 0.0)
    f1 = final_metrics.get("val_f1", 0.0)
    auc = final_metrics.get("val_auroc", final_metrics.get("val_auc", 0.0))
    loss = final_metrics.get("val_loss", 0.0)

    result = {
        "experiment_id": f"run_{run.id}",
        "status": run.status,
        "final_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "loss": loss,
        },
        "training_history": training_history,
        "total_epochs": len(training_history),
        "metadata": {
            "experiment_name": f"run_{run.id}",
            "start_time": run.start_time.isoformat() if run.start_time else "",
            "end_time": run.end_time.isoformat() if run.end_time else "",
            "total_epochs": len(training_history),
            "best_epoch": _find_best_epoch(training_history),
            "best_val_accuracy": max([h.get("val_acc", 0) for h in training_history], default=0.0),
            "best_val_recall": max([h.get("val_recall", 0) for h in training_history], default=0.0),
            "best_val_loss": min([h.get("val_loss", float('inf')) for h in training_history], default=0.0),
            "best_val_f1": max([h.get("val_f1", 0) for h in training_history], default=0.0),
            "best_val_auroc": max([h.get("val_auroc", 0) for h in training_history], default=0.0),
            # Include all final metrics in metadata for display
            "final_accuracy": accuracy,
            "final_precision": precision,
            "final_recall": recall,
            "final_f1": f1,
            "final_auc": auc,
            "final_loss": loss,
        },
        "confusion_matrix": None,  # TODO: Add if available in metrics
    }

    return result


def _find_best_epoch(training_history: List[Dict]) -> int:
    """Find epoch with best validation accuracy."""
    if not training_history:
        return 0

    best_epoch = 0
    best_acc = 0.0

    for entry in training_history:
        if entry.get("val_acc", 0) > best_acc:
            best_acc = entry["val_acc"]
            best_epoch = entry["epoch"]

    return best_epoch
