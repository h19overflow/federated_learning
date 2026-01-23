"""Endpoints for querying experiment status."""

import json
from pathlib import Path
from typing import Any, Dict

import aiofiles
from fastapi import APIRouter, HTTPException

from federated_pneumonia_detection.src.api.endpoints.experiments.utils import (
    calculate_progress,
    find_experiment_log_file,
)
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

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
                status_code=404,
                detail=f"Experiment not found: {experiment_id}",
            )

        async with aiofiles.open(log_file, "r") as f:
            content = await f.read()
            log_data = json.loads(content)

        metadata = log_data.get("metadata", {})
        epochs = log_data.get("epochs", [])
        current_epoch = log_data.get("current_epoch")

        progress = calculate_progress(log_data)

        latest_metrics = None
        for event in reversed(epochs):
            if event.get("type") == "epoch_end":
                latest_metrics = event.get("metrics", {})
                break

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
            status_code=500,
            detail=f"Failed to get experiment status: {str(e)}",
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
                async with aiofiles.open(log_file, "r") as f:
                    content = await f.read()
                    log_data = json.loads(content)

                metadata = log_data.get("metadata", {})
                progress = calculate_progress(log_data)

                # Extract experiment ID from filename or metadata
                experiment_id = log_file.stem

                experiments.append(
                    {
                        "experiment_id": experiment_id,
                        "experiment_name": metadata.get(
                            "experiment_name",
                            experiment_id,
                        ),
                        "status": metadata.get("status", "unknown"),
                        "training_mode": metadata.get("training_mode", "unknown"),
                        "progress": progress,
                        "current_epoch": log_data.get("current_epoch"),
                        "start_time": metadata.get("start_time"),
                        "end_time": metadata.get("end_time"),
                    },
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
            status_code=500,
            detail=f"Failed to list experiments: {str(e)}",
        )
