"""
Metric extraction logic for federated and centralized runs.

Provides independent metric extraction from server_evaluations (federated)
and run_metrics (centralized) tables.
"""

from typing import Optional, Tuple
import logging

from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import server_evaluation_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

logger = get_logger(__name__)


def extract_federated_metrics(
    db, run_id: int
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract best metrics from server_evaluations for federated run.

    Each metric (accuracy, precision, recall, f1_score) is extracted
    independently with its own null check.

    Args:
        db: Database session
        run_id: ID of the federated run

    Returns:
        Tuple of (accuracy, precision, recall, f1_score), each possibly None
    """
    logger.info(f"Processing federated run {run_id}")
    summary = server_evaluation_crud.get_summary_stats(db, run_id)
    logger.debug(f"Federated run {run_id} summary: {summary}")

    accuracy_value = None
    precision_value = None
    recall_value = None
    f1_value = None

    if summary:
        # Extract each metric independently with its own null check
        if summary.get('best_accuracy') and summary['best_accuracy'].get('value') is not None:
            accuracy_value = summary['best_accuracy']['value']
        if summary.get('best_precision') and summary['best_precision'].get('value') is not None:
            precision_value = summary['best_precision']['value']
        if summary.get('best_recall') and summary['best_recall'].get('value') is not None:
            recall_value = summary['best_recall']['value']
        if summary.get('best_f1_score') and summary['best_f1_score'].get('value') is not None:
            f1_value = summary['best_f1_score']['value']

    return accuracy_value, precision_value, recall_value, f1_value


def extract_centralized_metrics(
    db, run_id: int
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract best metrics from run_metrics for centralized run.

    Retrieves best (highest) values for accuracy, precision, recall, f1_score.

    Args:
        db: Database session
        run_id: ID of the centralized run

    Returns:
        Tuple of (accuracy, precision, recall, f1_score), each possibly None
    """
    best_accuracy = run_metric_crud.get_best_metric(db, run_id, "val_accuracy", maximize=True)
    best_precision = run_metric_crud.get_best_metric(db, run_id, "val_precision", maximize=True)
    best_recall = run_metric_crud.get_best_metric(db, run_id, "val_recall", maximize=True)
    best_f1 = run_metric_crud.get_best_metric(db, run_id, "val_f1", maximize=True)

    accuracy_value = best_accuracy.metric_value if best_accuracy else None
    precision_value = best_precision.metric_value if best_precision else None
    recall_value = best_recall.metric_value if best_recall else None
    f1_value = best_f1.metric_value if best_f1 else None

    return accuracy_value, precision_value, recall_value, f1_value
