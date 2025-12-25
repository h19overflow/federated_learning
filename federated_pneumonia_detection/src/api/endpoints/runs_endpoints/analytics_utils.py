"""
Utility functions for analytics aggregation.

Provides helper functions for calculating statistics and extracting
run details for analytics endpoints.
"""

from typing import Dict, Any, Optional, List
import logging

from federated_pneumonia_detection.src.boundary.engine import Run
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import server_evaluation_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

logger = get_logger(__name__)


def calculate_mode_statistics(db, runs: List[Run]) -> Dict[str, Any]:
    """
    Calculate aggregated statistics for a list of runs.

    Handles both centralized (run_metrics) and federated (server_evaluations) runs.

    Args:
        db: Database session
        runs: List of Run objects

    Returns:
        Dictionary with aggregated metrics (avg_accuracy, avg_precision, etc.)
    """
    if not runs:
        return {
            "count": 0,
            "avg_accuracy": None,
            "avg_precision": None,
            "avg_recall": None,
            "avg_f1": None,
            "avg_duration_minutes": None
        }

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    durations = []

    for run in runs:
        if run.training_mode == 'federated':
            # For federated runs, get best metrics from server_evaluations table
            logger.info(f"Processing federated run {run.id}")
            summary = server_evaluation_crud.get_summary_stats(db, run.id)
            logger.debug(f"Federated run {run.id} summary: {summary}")
            if summary and summary.get('best_accuracy'):
                if summary['best_accuracy'].get('value') is not None:
                    accuracies.append(summary['best_accuracy']['value'])
                if summary.get('best_precision') and summary['best_precision'].get('value') is not None:
                    precisions.append(summary['best_precision']['value'])
                if summary.get('best_recall') and summary['best_recall'].get('value') is not None:
                    recalls.append(summary['best_recall']['value'])
                if summary.get('best_f1_score') and summary['best_f1_score'].get('value') is not None:
                    f1_scores.append(summary['best_f1_score']['value'])
        else:
            # For centralized runs, get metrics from run_metrics table
            best_accuracy = run_metric_crud.get_best_metric(db, run.id, "val_accuracy", maximize=True)
            best_precision = run_metric_crud.get_best_metric(db, run.id, "val_precision", maximize=True)
            best_recall = run_metric_crud.get_best_metric(db, run.id, "val_recall", maximize=True)
            best_f1 = run_metric_crud.get_best_metric(db, run.id, "val_f1", maximize=True)

            if best_accuracy:
                accuracies.append(best_accuracy.metric_value)
            if best_precision:
                precisions.append(best_precision.metric_value)
            if best_recall:
                recalls.append(best_recall.metric_value)
            if best_f1:
                f1_scores.append(best_f1.metric_value)

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


def extract_run_details(db, run: Run) -> Optional[Dict[str, Any]]:
    """
    Extract detailed metrics for a single run.

    Handles both centralized and federated runs.

    Args:
        db: Database session
        run: Run object

    Returns:
        Dictionary with run details or None if metrics missing
    """
    try:
        accuracy_value = None
        precision_value = None
        recall_value = None
        f1_value = None

        if run.training_mode == 'federated':
            # For federated runs, get metrics from server_evaluations
            summary = server_evaluation_crud.get_summary_stats(db, run.id)
            if summary and summary.get('best_accuracy'):
                accuracy_value = summary['best_accuracy'].get('value')
                if summary.get('best_precision'):
                    precision_value = summary['best_precision'].get('value')
                if summary.get('best_recall'):
                    recall_value = summary['best_recall'].get('value')
                if summary.get('best_f1_score'):
                    f1_value = summary['best_f1_score'].get('value')
        else:
            # For centralized runs, get metrics from run_metrics
            best_accuracy = run_metric_crud.get_best_metric(db, run.id, "val_accuracy", maximize=True)
            best_precision = run_metric_crud.get_best_metric(db, run.id, "val_precision", maximize=True)
            best_recall = run_metric_crud.get_best_metric(db, run.id, "val_recall", maximize=True)
            best_f1 = run_metric_crud.get_best_metric(db, run.id, "val_f1", maximize=True)

            if best_accuracy:
                accuracy_value = best_accuracy.metric_value
            if best_precision:
                precision_value = best_precision.metric_value
            if best_recall:
                recall_value = best_recall.metric_value
            if best_f1:
                f1_value = best_f1.metric_value

        # Skip if no accuracy metric found
        if accuracy_value is None:
            logger.warning(f"No metrics found for run {run.id}, skipping from top runs")
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


def safe_average(values: List[float]) -> Optional[float]:
    """
    Calculate average of a list, handling empty lists gracefully.

    Args:
        values: List of numeric values

    Returns:
        Average rounded to 4 decimals or None if list is empty
    """
    if not values:
        return None
    return round(sum(values) / len(values), 4)


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
