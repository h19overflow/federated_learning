"""
Results service for managing experiment results retrieval and processing.

This module contains the business logic for loading, parsing, and formatting
experiment results from JSON files and artifact directories.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

logger = get_logger(__name__)

# Configure result directories
RESULTS_DIR = Path("results")
LOGS_DIR = Path("logs/progress")
CHECKPOINTS_DIR = Path("models/checkpoints")


def find_experiment_results_file(experiment_id: str) -> Optional[Path]:
    """Find the results JSON file for a given experiment ID."""
    if not RESULTS_DIR.exists():
        return None

    # Search for metrics JSON files containing the experiment ID
    for results_file in RESULTS_DIR.glob("*metrics*.json"):
        if experiment_id in results_file.stem:
            return results_file
    return None


def find_progress_log_file(experiment_id: str) -> Optional[Path]:
    """Find the progress log file for a given experiment ID."""
    if not LOGS_DIR.exists():
        return None

    for log_file in LOGS_DIR.glob("*.json"):
        if experiment_id in log_file.stem:
            return log_file
    return None


def parse_confusion_matrix_from_metrics(
    epoch_metrics: List[Dict],
) -> Optional[Dict[str, Any]]:
    """Extract confusion matrix from epoch metrics if available."""
    # Look for confusion matrix values in the last epoch
    if not epoch_metrics:
        return None

    last_epoch = epoch_metrics[-1]
    cm_data = {}

    # Check for common confusion matrix metric names
    for key in [
        "val_true_positives",
        "val_true_negatives",
        "val_false_positives",
        "val_false_negatives",
        "TP",
        "TN",
        "FP",
        "FN",
    ]:
        if key.lower() in str(last_epoch).lower():
            cm_data[key] = last_epoch.get(key, 0)

    if cm_data:
        return {
            "true_positives": cm_data.get("val_true_positives", cm_data.get("TP", 0)),
            "true_negatives": cm_data.get("val_true_negatives", cm_data.get("TN", 0)),
            "false_positives": cm_data.get("val_false_positives", cm_data.get("FP", 0)),
            "false_negatives": cm_data.get("val_false_negatives", cm_data.get("FN", 0)),
        }
    return None


def extract_final_metrics(last_epoch: Dict[str, Any]) -> Dict[str, Any]:
    """Extract final metrics from the last training epoch."""
    return {
        "accuracy": last_epoch.get("val_acc", last_epoch.get("val_accuracy", 0)),
        "precision": last_epoch.get("val_precision", 0),
        "recall": last_epoch.get("val_recall", 0),
        "f1_score": last_epoch.get("val_f1", 0),
        "auc": last_epoch.get("val_auroc", last_epoch.get("val_auc", 0)),
        "loss": last_epoch.get("val_loss", 0),
    }


def build_training_history(epoch_metrics: List[Dict]) -> List[Dict[str, Any]]:
    """Build training history from epoch metrics."""
    training_history = []
    for epoch_data in epoch_metrics:
        training_history.append(
            {
                "epoch": epoch_data.get("epoch", 0),
                "train_loss": epoch_data.get("train_loss", 0),
                "val_loss": epoch_data.get("val_loss", 0),
                "train_acc": epoch_data.get(
                    "train_acc", epoch_data.get("train_accuracy", 0)
                ),
                "val_acc": epoch_data.get(
                    "val_acc", epoch_data.get("val_accuracy", 0)
                ),
            }
        )
    return training_history


def load_results_file(results_file: Path) -> Dict[str, Any]:
    """Load and parse a results JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def get_experiment_results(experiment_id: str) -> Dict[str, Any]:
    """
    Retrieve complete results for a specific experiment.

    Args:
        experiment_id: Unique identifier for the experiment

    Returns:
        Dictionary containing experiment results including metrics and artifacts
    """
    # Find results file
    results_file = find_experiment_results_file(experiment_id)
    if not results_file:
        return None

    # Load results
    results_data = load_results_file(results_file)

    metadata = results_data.get("metadata", {})
    epoch_metrics = results_data.get("epoch_metrics", [])

    # Get final metrics from last epoch
    final_metrics = {}
    if epoch_metrics:
        final_metrics = extract_final_metrics(epoch_metrics[-1])

    # Extract confusion matrix if available
    confusion_matrix = parse_confusion_matrix_from_metrics(epoch_metrics)

    # Get training history for charts
    training_history = build_training_history(epoch_metrics)

    return {
        "experiment_id": experiment_id,
        "status": "completed",
        "metadata": metadata,
        "final_metrics": final_metrics,
        "confusion_matrix": confusion_matrix,
        "training_history": training_history,
        "total_epochs": len(epoch_metrics),
    }


def get_experiment_metrics(experiment_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve training metrics for a specific experiment.

    Args:
        experiment_id: Unique identifier for the experiment

    Returns:
        Dictionary containing training metrics (accuracy, loss, etc.)
    """
    results_file = find_experiment_results_file(experiment_id)
    if not results_file:
        return None

    results_data = load_results_file(results_file)

    epoch_metrics = results_data.get("epoch_metrics", [])
    metadata = results_data.get("metadata", {})

    # Get final metrics from last epoch
    final_metrics = {}
    if epoch_metrics:
        final_metrics = extract_final_metrics(epoch_metrics[-1])

    return {
        "experiment_id": experiment_id,
        "final_metrics": final_metrics,
        "best_metrics": {
            "best_val_recall": metadata.get("best_val_recall", 0),
            "best_val_loss": metadata.get("best_val_loss", float("inf")),
            "best_epoch": metadata.get("best_epoch", 0),
        },
        "total_epochs": len(epoch_metrics),
    }


def get_experiment_artifacts(
    experiment_id: str, artifact_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve artifacts for a specific experiment (models, checkpoints, etc.).

    Args:
        experiment_id: Unique identifier for the experiment
        artifact_type: Optional filter for specific artifact types

    Returns:
        List of artifact dictionaries
    """
    artifacts = []

    # Find metrics JSON files
    if artifact_type is None or artifact_type == "metrics":
        if RESULTS_DIR.exists():
            for file in RESULTS_DIR.glob(f"*{experiment_id}*"):
                if file.is_file():
                    artifacts.append(
                        {
                            "type": "metrics",
                            "filename": file.name,
                            "path": str(file),
                            "size_bytes": file.stat().st_size,
                            "format": file.suffix[1:],  # Remove the dot
                        }
                    )

    # Find checkpoint files
    if artifact_type is None or artifact_type == "checkpoint":
        if CHECKPOINTS_DIR.exists():
            for file in CHECKPOINTS_DIR.glob(f"*{experiment_id}*.ckpt"):
                artifacts.append(
                    {
                        "type": "checkpoint",
                        "filename": file.name,
                        "path": str(file),
                        "size_bytes": file.stat().st_size,
                        "format": "ckpt",
                    }
                )

    # Find progress logs
    if artifact_type is None or artifact_type == "logs":
        log_file = find_progress_log_file(experiment_id)
        if log_file:
            artifacts.append(
                {
                    "type": "logs",
                    "filename": log_file.name,
                    "path": str(log_file),
                    "size_bytes": log_file.stat().st_size,
                    "format": "json",
                }
            )

    return artifacts


def list_experiments(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available experiments with optional filtering by status.

    Args:
        status: Optional status filter (completed, running, failed)

    Returns:
        List of experiment dictionaries
    """
    experiments = []

    # Scan results directory for completed experiments
    if RESULTS_DIR.exists():
        for results_file in RESULTS_DIR.glob("*metrics*.json"):
            try:
                data = load_results_file(results_file)

                metadata = data.get("metadata", {})
                experiment_name = metadata.get("experiment_name", results_file.stem)

                # Determine status (simplified)
                exp_status = "completed" if metadata.get("end_time") else "running"

                # Apply status filter if provided
                if status and exp_status != status:
                    continue

                experiments.append(
                    {
                        "experiment_id": experiment_name,
                        "experiment_name": experiment_name,
                        "status": exp_status,
                        "start_time": metadata.get("start_time"),
                        "end_time": metadata.get("end_time"),
                        "total_epochs": metadata.get("total_epochs", 0),
                        "best_val_recall": metadata.get("best_val_recall", 0),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not parse {results_file}: {e}")
                continue

    return experiments


def find_artifact_by_type(
    experiment_id: str, artifact_type: str
) -> Optional[Path]:
    """
    Find a specific artifact file for an experiment by type.

    Args:
        experiment_id: Unique identifier for the experiment
        artifact_type: Type of artifact to find

    Returns:
        Path to the artifact file if found, None otherwise
    """
    if artifact_type == "metrics_json":
        results_file = find_experiment_results_file(experiment_id)
        if results_file and results_file.suffix == ".json":
            return results_file

    elif artifact_type == "metrics_csv":
        if RESULTS_DIR.exists():
            for file in RESULTS_DIR.glob(f"*{experiment_id}*.csv"):
                return file

    elif artifact_type == "summary":
        if RESULTS_DIR.exists():
            for file in RESULTS_DIR.glob(f"*{experiment_id}*summary*.txt"):
                return file

    elif artifact_type == "logs":
        return find_progress_log_file(experiment_id)

    elif artifact_type == "checkpoint":
        if CHECKPOINTS_DIR.exists():
            for file in CHECKPOINTS_DIR.glob(f"*{experiment_id}*.ckpt"):
                return file

    return None


def build_comparison_results(
    centralized_file: Optional[Path],
    federated_file: Optional[Path],
) -> tuple[Optional[Dict], Optional[Dict], Dict]:
    """
    Build comparison results from centralized and federated results files.

    Args:
        centralized_file: Path to centralized results file
        federated_file: Path to federated results file

    Returns:
        Tuple of (centralized_results, federated_results, comparison_metrics)
    """
    centralized_results = None
    federated_results = None
    comparison_metrics = {}

    # Load centralized results
    if centralized_file:
        with open(centralized_file, "r") as f:
            cent_data = json.load(f)
        cent_epochs = cent_data.get("epoch_metrics", [])
        if cent_epochs:
            last_epoch = cent_epochs[-1]
            centralized_results = {
                "accuracy": last_epoch.get(
                    "val_acc", last_epoch.get("val_accuracy", 0)
                ),
                "precision": last_epoch.get("val_precision", 0),
                "recall": last_epoch.get("val_recall", 0),
                "f1_score": last_epoch.get("val_f1", 0),
                "auc": last_epoch.get("val_auroc", last_epoch.get("val_auc", 0)),
                "loss": last_epoch.get("val_loss", 0),
                "training_history": [
                    {
                        "epoch": ep.get("epoch", 0),
                        "train_loss": ep.get("train_loss", 0),
                        "val_loss": ep.get("val_loss", 0),
                        "train_acc": ep.get(
                            "train_acc", ep.get("train_accuracy", 0)
                        ),
                        "val_acc": ep.get("val_acc", ep.get("val_accuracy", 0)),
                    }
                    for ep in cent_epochs
                ],
            }

    # Load federated results
    if federated_file:
        with open(federated_file, "r") as f:
            fed_data = json.load(f)
        fed_epochs = fed_data.get("epoch_metrics", [])
        if fed_epochs:
            last_epoch = fed_epochs[-1]
            federated_results = {
                "accuracy": last_epoch.get(
                    "val_acc", last_epoch.get("val_accuracy", 0)
                ),
                "precision": last_epoch.get("val_precision", 0),
                "recall": last_epoch.get("val_recall", 0),
                "f1_score": last_epoch.get("val_f1", 0),
                "auc": last_epoch.get("val_auroc", last_epoch.get("val_auc", 0)),
                "loss": last_epoch.get("val_loss", 0),
                "training_history": [
                    {
                        "epoch": ep.get("epoch", 0),
                        "train_loss": ep.get("train_loss", 0),
                        "val_loss": ep.get("val_loss", 0),
                        "train_acc": ep.get(
                            "train_acc", ep.get("train_accuracy", 0)
                        ),
                        "val_acc": ep.get("val_acc", ep.get("val_accuracy", 0)),
                    }
                    for ep in fed_epochs
                ],
            }

    # Calculate comparison metrics
    if centralized_results and federated_results:
        for metric in ["accuracy", "precision", "recall", "f1_score", "auc"]:
            cent_val = centralized_results.get(metric, 0)
            fed_val = federated_results.get(metric, 0)
            comparison_metrics[metric] = {
                "centralized": cent_val,
                "federated": fed_val,
                "difference": cent_val - fed_val,
                "percent_difference": ((cent_val - fed_val) / cent_val * 100)
                if cent_val != 0
                else 0,
            }

    return centralized_results, federated_results, comparison_metrics
