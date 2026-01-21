"""
Performance metrics calculation for benchmark results.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    true_labels: List[int],
    pred_labels: List[int],
    pred_probs: List[float] = None,
) -> Dict[str, float]:
    """
    Calculate classification metrics.

    Args:
        true_labels: True binary labels (0=Normal, 1=Pneumonia)
        pred_labels: Predicted binary labels
        pred_probs: Predicted probabilities for positive class

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    try:
        metrics["accuracy"] = accuracy_score(true_labels, pred_labels)
        metrics["precision"] = precision_score(
            true_labels,
            pred_labels,
            zero_division=0,
        )
        metrics["recall"] = recall_score(true_labels, pred_labels, zero_division=0)
        metrics["f1"] = f1_score(true_labels, pred_labels, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics["sensitivity"] = recall_score(
                true_labels,
                pred_labels,
                zero_division=0,
            )

        # AUC-ROC if probabilities provided
        if pred_probs is not None:
            metrics["auroc"] = roc_auc_score(true_labels, pred_probs)

    except Exception as e:
        import logging

        logging.warning(f"Failed to calculate metrics: {e}")

    return metrics


def calculate_stage_statistics(timings: List[float]) -> Dict[str, float]:
    """
    Calculate statistical metrics for timing data.

    Args:
        timings: List of timing values in milliseconds

    Returns:
        Dictionary of statistics
    """
    if not timings:
        return {}

    return {
        "mean": np.mean(timings),
        "median": np.median(timings),
        "p50": np.percentile(timings, 50),
        "p95": np.percentile(timings, 95),
        "p99": np.percentile(timings, 99),
        "min": np.min(timings),
        "max": np.max(timings),
        "stddev": np.std(timings),
        "count": len(timings),
    }
