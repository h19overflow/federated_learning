"""
Endpoints for querying experiment status.

Poll-based status updates for training experiments with progress tracking.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pathlib import Path
import json

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from federated_pneumonia_detection.src.api.endpoints.experiments.utils import (
    find_experiment_log_file,
    calculate_progress,
)

router = APIRouter(
    prefix="/experiments",
    tags=["experiments", "status"],
)

logger = get_logger(__name__)

# Default log directory
LOGS_DIR = Path("logs/progress")


@router.get("/status/{experiment_id}")
async def get_experiment_status(experiment_id: str) -> Dict[str, Any]:
    """
    Get training experiment status.

    Parameters:
        experiment_id: Unique experiment identifier

    Returns:
        Experiment status including progress, current/total epochs,
        start/end times, and latest metrics.
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
    List all experiments with current status.

    Returns:
        List of experiments from logs directory with status,
        progress, and metadata, sorted by start time (most recent first).
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
