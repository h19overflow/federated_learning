"""
Utility functions for extracting and formatting run details.

For statistics aggregation, use calculate_mode_statistics from
analytics_utils_pkg.aggregators.
"""

from typing import Dict, Any, Optional
import logging

from federated_pneumonia_detection.src.boundary.engine import Run
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from .analytics_utils_pkg.extractors import extract_federated_metrics, extract_centralized_metrics

logger = get_logger(__name__)


def extract_run_details(db, run: Run) -> Optional[Dict[str, Any]]:
    """
    Extract detailed metrics for a single run.

    Handles both centralized and federated runs. Returns None if no
    accuracy metric is found.

    Args:
        db: Database session
        run: Run object

    Returns:
        Dictionary with run details or None if metrics missing
    """
    try:
        if run.training_mode == 'federated':
            accuracy_value, precision_value, recall_value, f1_value = extract_federated_metrics(db, run.id)
        else:
            accuracy_value, precision_value, recall_value, f1_value = extract_centralized_metrics(db, run.id)

        # Skip if no accuracy metric found
        if accuracy_value is None:
            logger.warning(f"No metrics found for run {run.id}, skipping")
            return None

        # Calculate duration
        duration_minutes = None
        if run.start_time and run.end_time:
            duration_minutes = round((run.end_time - run.start_time).total_seconds() / 60, 2)

        return {
            "run_id": run.id,
            "training_mode": run.training_mode,
            "best_accuracy": round(accuracy_value, 4) if accuracy_value is not None else None,
            "best_precision": round(precision_value, 4) if precision_value is not None else None,
            "best_recall": round(recall_value, 4) if recall_value is not None else None,
            "best_f1": round(f1_value, 4) if f1_value is not None else None,
            "duration_minutes": duration_minutes,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "status": run.status
        }
    except Exception as e:
        logger.error(f"Error extracting details for run {run.id}: {e}")
        return None


def create_empty_response() -> Dict[str, Any]:
    """Generate empty response structure when no runs found."""
    empty_stats = {
        "count": 0,
        "avg_accuracy": None,
        "avg_precision": None,
        "avg_recall": None,
        "avg_f1": None,
        "avg_duration_minutes": None
    }
    return {
        "total_runs": 0,
        "success_rate": 0.0,
        "centralized": empty_stats,
        "federated": empty_stats,
        "top_runs": []
    }
