"""
Metric extraction logic for federated and centralized runs.

Provides functions to extract best metrics from different data sources
(server_evaluations for federated, run_metrics for centralized).
"""

from typing import Dict, Any, Optional, Tuple
import logging

from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import server_evaluation_crud
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

logger = get_logger(__name__)


def extract_federated_metrics(
    db,
    run_id: int
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract best metrics from server_evaluations table for federated run.

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
        if summary.get('best_accuracy'):
            accuracy_value = summary['best_accuracy'].get('value')
        if summary.get('best_precision'):
            precision_value = summary['best_precision'].get('value')
        if summary.get('best_recall'):
            recall_value = summary['best_recall'].get('value')
        if summary.get('best_f1_score'):
            f1_value = summary['best_f1_score'].get('value')

    return accuracy_value, precision_value, recall_value, f1_value


def extract_centralized_metrics(
    db,
    run_id: int
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract best metrics from run_metrics table for centralized run.

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
