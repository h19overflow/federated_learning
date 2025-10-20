"""
Endpoints for querying experiment status.

This module provides HTTP endpoints to query the current status of training
experiments, including progress, current epoch, and overall completion status.
Useful for polling-based status updates when WebSocket isn't available.

Dependencies:
- pathlib: File path operations
- json: Log file parsing
- fastapi: HTTP endpoint framework
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pathlib import Path
import json

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

router = APIRouter(
    prefix="/experiments",
    tags=["experiments", "status"],
)

logger = get_logger(__name__)

# Default log directory
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
    metadata = log_data.get("metadata", {})
    current_epoch = log_data.get("current_epoch")
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


@router.get("/status/{experiment_id}")
async def get_experiment_status(experiment_id: str) -> Dict[str, Any]:
    """
    Get the current status of a training experiment.

    Returns real-time status information including current progress,
    epoch information, and overall training status.

    **Parameters:**
    - `experiment_id`: Unique experiment identifier

    **Response:**
    ```json
    {
        "experiment_id": "exp123",
        "status": "running",
        "progress": 45.5,
        "current_epoch": 5,
        "total_epochs": 10,
        "training_mode": "centralized",
        "start_time": "2025-10-20T10:00:00",
        "latest_metrics": {...}
    }
    ```

    **Status Values:**
    - `started`: Training initialized
    - `running`: Training in progress
    - `completed`: Training finished successfully
    - `failed`: Training encountered an error
    """
    try:
        log_file = find_experiment_log_file(experiment_id)

        if not log_file:
            raise HTTPException(
                status_code=404, detail=f"Experiment not found: {experiment_id}"
            )

        # Read log file
        with open(log_file, "r") as f:
            log_data = json.load(f)

        metadata = log_data.get("metadata", {})
        epochs = log_data.get("epochs", [])
        current_epoch = log_data.get("current_epoch")

        # Calculate progress
        progress = calculate_progress(log_data)

        # Get latest metrics from most recent epoch_end event
        latest_metrics = None
        for event in reversed(epochs):
            if event.get("type") == "epoch_end":
                latest_metrics = event.get("metrics", {})
                break

        # Determine total epochs
        total_epochs = None
        for event in epochs:
            if event.get("type") == "epoch_start" and "total_epochs" in event:
                total_epochs = event["total_epochs"]
                break

        return {
            "experiment_id": experiment_id,
            "experiment_name": metadata.get("experiment_name", experiment_id),
            "status": metadata.get("status", "unknown"),
            "training_mode": metadata.get("training_mode", "unknown"),
            "progress": progress,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "start_time": metadata.get("start_time"),
            "end_time": metadata.get("end_time"),
            "latest_metrics": latest_metrics,
            "total_events": len(epochs),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for experiment {experiment_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get experiment status: {str(e)}"
        )


@router.get("/list")
async def list_experiments() -> Dict[str, Any]:
    """
    List all available experiments with their current status.

    Returns a list of all experiments found in the logs directory,
    including their current status and basic information.

    **Response:**
    ```json
    {
        "experiments": [
            {
                "experiment_id": "exp123",
                "experiment_name": "pneumonia_cent_2025-10-20",
                "status": "completed",
                "training_mode": "centralized",
                "progress": 100.0,
                "start_time": "2025-10-20T10:00:00"
            }
        ],
        "count": 1
    }
    ```
    """
    try:
        if not LOGS_DIR.exists():
            return {
                "experiments": [],
                "count": 0,
            }

        experiments = []
        for log_file in LOGS_DIR.glob("*.json"):
            try:
                with open(log_file, "r") as f:
                    log_data = json.load(f)

                metadata = log_data.get("metadata", {})
                progress = calculate_progress(log_data)

                # Extract experiment ID from filename or metadata
                experiment_id = log_file.stem

                experiments.append(
                    {
                        "experiment_id": experiment_id,
                        "experiment_name": metadata.get(
                            "experiment_name", experiment_id
                        ),
                        "status": metadata.get("status", "unknown"),
                        "training_mode": metadata.get("training_mode", "unknown"),
                        "progress": progress,
                        "current_epoch": log_data.get("current_epoch"),
                        "start_time": metadata.get("start_time"),
                        "end_time": metadata.get("end_time"),
                    }
                )
            except Exception as e:
                logger.warning(f"Error reading experiment log {log_file}: {str(e)}")
                continue

        # Sort by start time (most recent first)
        experiments.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        return {
            "experiments": experiments,
            "count": len(experiments),
        }

    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list experiments: {str(e)}"
        )
