"""
Statistics aggregation for training runs.

Calculates average metrics across multiple runs by training mode
(centralized vs federated).
"""

from typing import Dict, Any, Optional, List
import logging

from federated_pneumonia_detection.src.boundary.engine import Run
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from .extractors import extract_federated_metrics, extract_centralized_metrics

logger = get_logger(__name__)


def calculate_mode_statistics(db, runs: List[Run]) -> Dict[str, Any]:
    """
    Calculate aggregated statistics for a list of runs.

    Handles both centralized (run_metrics) and federated (server_evaluations).
    Returns averages for accuracy, precision, recall, f1, and duration.

    Args:
        db: Database session
        runs: List of Run objects

    Returns:
        Dictionary with aggregated metrics and count
    """
    if not runs:
        return _empty_statistics()

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    durations = []

    for run in runs:
        if run.training_mode == 'federated':
            accuracy, precision, recall, f1_score = extract_federated_metrics(db, run.id)
        else:
            accuracy, precision, recall, f1_score = extract_centralized_metrics(db, run.id)

        # Append metrics if available (independent of each other)
        if accuracy is not None:
            accuracies.append(accuracy)
        if precision is not None:
            precisions.append(precision)
        if recall is not None:
            recalls.append(recall)
        if f1_score is not None:
            f1_scores.append(f1_score)

        # Calculate duration (same for both modes)
        if run.start_time and run.end_time:
            duration = (run.end_time - run.start_time).total_seconds() / 60
            durations.append(duration)

    return {
        "count": len(runs),
        "avg_accuracy": safe_average(accuracies),
        "avg_precision": safe_average(precisions),
        "avg_recall": safe_average(recalls),
        "avg_f1": safe_average(f1_scores),
        "avg_duration_minutes": safe_average(durations)
    }


def _empty_statistics() -> Dict[str, Any]:
    """Generate empty statistics dictionary."""
    return {
        "count": 0,
        "avg_accuracy": None,
        "avg_precision": None,
        "avg_recall": None,
        "avg_f1": None,
        "avg_duration_minutes": None
    }


def safe_average(values: List[float]) -> Optional[float]:
    """
    Calculate average of a list gracefully.

    Args:
        values: List of numeric values

    Returns:
        Average rounded to 4 decimals or None if list is empty
    """
    if not values:
        return None
    return round(sum(values) / len(values), 4)
