"""
Status and logging utilities for experiment endpoints.

Provides functions for finding experiment logs and calculating training progress.
"""

from pathlib import Path
from typing import Any, Dict, Optional

LOGS_DIR = Path("logs/progress")


def find_experiment_log_file(experiment_id: str) -> Optional[Path]:
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
