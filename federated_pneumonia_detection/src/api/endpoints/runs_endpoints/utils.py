"""
Utility functions for transforming run data to API response format.

Converts database Run objects to frontend-compatible ExperimentResults format,
including epoch indexing conversion, metrics aggregation, and summary statistics.
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict


def _calculate_summary_statistics(cm: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate derived metrics from confusion matrix values.

    Args:
        cm: Dict with keys: true_positives, true_negatives, false_positives, false_negatives

    Returns:
        Dict with calculated statistics: sensitivity, specificity, precision_cm, accuracy_cm, f1_cm

    Raises:
        ValueError: If any confusion matrix value is negative or total samples is zero
    """
    tp = cm.get("true_positives", 0)
    tn = cm.get("true_negatives", 0)
    fp = cm.get("false_positives", 0)
    fn = cm.get("false_negatives", 0)

    # Validate values
    if any(v < 0 for v in [tp, tn, fp, fn]):
        raise ValueError("Confusion matrix values cannot be negative")

    total = tp + tn + fp + fn
    if total == 0:
        return {
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision_cm": 0.0,
            "accuracy_cm": 0.0,
            "f1_cm": 0.0,
        }

    # Sensitivity (Recall for positive class) = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (Recall for negative class) = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision = TP / (TP + FP)
    precision_cm = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Accuracy = (TP + TN) / Total
    accuracy_cm = (tp + tn) / total if total > 0 else 0.0

    # F1 Score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    denom = precision_cm + sensitivity
    f1_cm = 2 * (precision_cm * sensitivity) / denom if denom > 0 else 0.0

    return {
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision_cm": round(precision_cm, 4),
        "accuracy_cm": round(accuracy_cm, 4),
        "f1_cm": round(f1_cm, 4),
    }


def _transform_run_to_results(run) -> Dict[str, Any]:
    """
    Transform database Run object to ExperimentResults format.

    Database format:
        RunMetric(metric_name='val_recall', metric_value=0.95, step=10, dataset_type='validation')
        Note: step is 0-indexed (0-9 for 10 epochs)

    Frontend format:
        {
          final_metrics: {accuracy: 0.92, ...},
          training_history: [{epoch: 1, train_loss: 0.5, ...}],  # epoch is 1-indexed (1-10)
          ...
        }

    Important: Converts epochs from 0-indexed (database) to 1-indexed (display)
    """

    # Group metrics by epoch (step)
    metrics_by_epoch = defaultdict(dict)
    final_metrics = {}

    for metric in run.metrics:
        epoch = metric.step
        metric_name = metric.metric_name
        value = metric.metric_value

        # Store in epoch-based structure
        metrics_by_epoch[epoch][metric_name] = value

        # Track final (last epoch) metrics
        if epoch >= max(metrics_by_epoch.keys(), default=0):
            if metric_name in ['val_accuracy', 'val_acc', 'val_precision', 'val_recall',
                              'val_f1', 'val_auroc', 'val_auc', 'val_loss']:
                final_metrics[metric_name] = value

    # Build training history
    training_history = []
    for epoch in sorted(metrics_by_epoch.keys()):
        epoch_data = metrics_by_epoch[epoch]

        training_history.append({
            "epoch": epoch + 1,  # Convert from 0-indexed to 1-indexed for display
            "train_loss": epoch_data.get("train_loss", 0.0),
            "val_loss": epoch_data.get("val_loss", 0.0),
            "train_acc": epoch_data.get("train_accuracy", epoch_data.get("train_acc", 0.0)),
            "val_acc": epoch_data.get("val_accuracy", epoch_data.get("val_acc", 0.0)),
            "train_f1": epoch_data.get("train_f1", 0.0),
            "val_precision": epoch_data.get("val_precision", 0.0),
            "val_recall": epoch_data.get("val_recall", 0.0),
            "val_f1": epoch_data.get("val_f1", 0.0),
            "val_auroc": epoch_data.get("val_auroc", epoch_data.get("val_auc", 0.0)),
        })

    # Extract final metrics (use last epoch values or best values)
    last_epoch_data = metrics_by_epoch[max(metrics_by_epoch.keys())] if metrics_by_epoch else {}

    # Calculate final metrics from last epoch
    accuracy = final_metrics.get("val_accuracy", final_metrics.get("val_acc", 0.0))
    precision = final_metrics.get("val_precision", 0.0)
    recall = final_metrics.get("val_recall", 0.0)
    f1 = final_metrics.get("val_f1", 0.0)
    auc = final_metrics.get("val_auroc", final_metrics.get("val_auc", 0.0))
    loss = final_metrics.get("val_loss", 0.0)

    # Extract confusion matrix values from last epoch
    confusion_matrix_obj = None
    if last_epoch_data:
        tp = last_epoch_data.get("val_cm_tp")
        tn = last_epoch_data.get("val_cm_tn")
        fp = last_epoch_data.get("val_cm_fp")
        fn = last_epoch_data.get("val_cm_fn")

        # Only build CM object if all values exist
        if all(v is not None for v in [tp, tn, fp, fn]):
            try:
                cm_dict = {
                    "true_positives": int(tp),
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                }
                # Calculate summary statistics
                summary_stats = _calculate_summary_statistics(cm_dict)
                confusion_matrix_obj = {**cm_dict, **summary_stats}
            except (ValueError, TypeError):
                confusion_matrix_obj = None

    result = {
        "experiment_id": f"run_{run.id}",
        "status": run.status,
        "final_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "loss": loss,
        },
        "training_history": training_history,
        "total_epochs": len(training_history),
        "metadata": {
            "experiment_name": f"run_{run.id}",
            "start_time": run.start_time.isoformat() if run.start_time else "",
            "end_time": run.end_time.isoformat() if run.end_time else "",
            "total_epochs": len(training_history),
            "best_epoch": _find_best_epoch(training_history),
            "best_val_accuracy": max([h.get("val_acc", 0) for h in training_history], default=0.0),
            "best_val_precision": max([h.get("val_precision", 0) for h in training_history], default=0.0),
            "best_val_recall": max([h.get("val_recall", 0) for h in training_history], default=0.0),
            "best_val_loss": min([h.get("val_loss", float('inf')) for h in training_history], default=0.0),
            "best_val_f1": max([h.get("val_f1", 0) for h in training_history], default=0.0),
            "best_val_auroc": max([h.get("val_auroc", 0) for h in training_history], default=0.0),
            # Include all final metrics in metadata for display
            "final_accuracy": accuracy,
            "final_precision": precision,
            "final_recall": recall,
            "final_f1": f1,
            "final_auc": auc,
            "final_loss": loss,
        },
        "confusion_matrix": confusion_matrix_obj,
    }

    return result


def _find_best_epoch(training_history: List[Dict]) -> int:
    """
    Find epoch with best validation accuracy.

    Note: Expects training_history with 1-indexed epochs (already transformed).
    Returns 1 as minimum since epochs are now displayed as 1-10 instead of 0-9.
    """
    if not training_history:
        return 1  # Return 1 instead of 0 since epochs are now 1-indexed

    best_epoch = 1
    best_acc = 0.0

    for entry in training_history:
        if entry.get("val_acc", 0) > best_acc:
            best_acc = entry["val_acc"]
            best_epoch = entry["epoch"]  # This is already 1-indexed from transformation

    return best_epoch
