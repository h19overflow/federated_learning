"""Transformation utilities for converting run data to analytics format.

This module contains functions for:
- Transforming database Run objects to ExperimentResults format
- Calculating summary statistics from confusion matrices
- Finding best epochs from training history
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional


def calculate_summary_statistics(cm: Dict[str, int]) -> Dict[str, float]:
    """Calculate derived metrics from confusion matrix values.

    Args:
        cm: Dict with keys: true_positives, true_negatives, false_positives, false_negatives  # noqa: E501

    Returns:
        Dict with calculated statistics: sensitivity, specificity, precision_cm, accuracy_cm, f1_cm  # noqa: E501

    Raises:
        ValueError: If any confusion matrix value is negative or total samples is zero
    """
    tp = cm.get("true_positives", 0)
    tn = cm.get("true_negatives", 0)
    fp = cm.get("false_positives", 0)
    fn = cm.get("false_negatives", 0)

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

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision_cm = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy_cm = (tp + tn) / total if total > 0 else 0.0

    denom = precision_cm + sensitivity
    f1_cm = 2 * (precision_cm * sensitivity) / denom if denom > 0 else 0.0

    return {
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision_cm": round(precision_cm, 4),
        "accuracy_cm": round(accuracy_cm, 4),
        "f1_cm": round(f1_cm, 4),
    }


def transform_run_to_results(
    run, persisted_stats: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Transform database Run object to ExperimentResults format.

    Args:
        run: Database Run object with related metrics
        persisted_stats: Optional pre-computed final epoch stats to use instead of calculating  # noqa: E501

    Returns:
        Dictionary containing transformed run data with training history, final metrics,
        and confusion matrix information.
    """
    is_federated = run.training_mode == "federated"
    metrics_by_epoch = defaultdict(dict)

    if is_federated:
        # For federated runs, read from server_evaluations
        for eval in run.server_evaluations:
            round_num = eval.round_number
            # Map server evaluation fields to training history format
            # Use 0.0 default for nullable metrics to prevent None comparison errors
            metrics_by_epoch[round_num] = {
                "train_loss": 0.0,  # Not available in server evaluations
                "val_loss": eval.loss if eval.loss is not None else 0.0,
                "train_accuracy": 0.0,  # Not available
                "train_acc": 0.0,  # Not available
                "val_accuracy": eval.accuracy if eval.accuracy is not None else 0.0,
                "val_acc": eval.accuracy if eval.accuracy is not None else 0.0,
                "train_f1": 0.0,  # Not available
                "val_precision": eval.precision if eval.precision is not None else 0.0,
                "val_recall": eval.recall if eval.recall is not None else 0.0,
                "val_f1": eval.f1_score if eval.f1_score is not None else 0.0,
                "val_auroc": eval.auroc if eval.auroc is not None else 0.0,
                "val_auc": eval.auroc
                if eval.auroc is not None
                else 0.0,  # Alternate name
                # Confusion matrix values
                "val_cm_tp": eval.true_positives,
                "val_cm_tn": eval.true_negatives,
                "val_cm_fp": eval.false_positives,
                "val_cm_fn": eval.false_negatives,
            }
    else:
        # For centralized runs, read from RunMetric table
        for metric in run.metrics:
            epoch = metric.step
            metric_name = metric.metric_name
            value = metric.metric_value

            # Store in epoch-based structure
            metrics_by_epoch[epoch][metric_name] = value

    # Build training history after metrics are fully populated
    training_history = []
    for epoch in sorted(metrics_by_epoch.keys()):
        epoch_data = metrics_by_epoch[epoch]

        # Convert epoch to 1-indexed display format
        # Centralized: step is 0-indexed (0-9) → add 1 for display (1-10)
        # Federated: round_number is already 1-indexed → use as-is
        display_epoch = epoch + 1 if not is_federated else epoch

        training_history.append(
            {
                "epoch": display_epoch,
                "train_loss": epoch_data.get("train_loss") or 0.0,
                "val_loss": epoch_data.get("val_loss") or 0.0,
                "train_acc": epoch_data.get("train_accuracy")
                or epoch_data.get("train_acc")
                or 0.0,
                "val_acc": epoch_data.get("val_accuracy")
                or epoch_data.get("val_acc")
                or 0.0,
                "train_f1": epoch_data.get("train_f1") or 0.0,
                "val_precision": epoch_data.get("val_precision") or 0.0,
                "val_recall": epoch_data.get("val_recall") or 0.0,
                "val_f1": epoch_data.get("val_f1") or 0.0,
                "val_auroc": epoch_data.get("val_auroc")
                or epoch_data.get("val_auc")
                or 0.0,
            },
        )

    # Extract final metrics from the LAST epoch (highest epoch number)
    # IMPORTANT: Do this AFTER metrics_by_epoch is fully populated to avoid out-of-order issues  # noqa: E501
    last_epoch_data = (
        metrics_by_epoch[max(metrics_by_epoch.keys())] if metrics_by_epoch else {}
    )

    # Calculate final metrics from last epoch data (with None handling)
    accuracy = (
        last_epoch_data.get("val_accuracy") or last_epoch_data.get("val_acc") or 0.0
    )
    precision = last_epoch_data.get("val_precision") or 0.0
    recall = last_epoch_data.get("val_recall") or 0.0
    f1 = last_epoch_data.get("val_f1") or 0.0
    auc = last_epoch_data.get("val_auroc") or last_epoch_data.get("val_auc") or 0.0
    loss = last_epoch_data.get("val_loss") or 0.0

    # Extract confusion matrix values from last epoch
    # Use persisted stats if available, otherwise calculate on-the-fly
    confusion_matrix_obj = None
    if persisted_stats:
        if last_epoch_data:
            # Use pre-computed stats but also include raw CM values from last epoch
            tp = last_epoch_data.get("val_cm_tp")
            tn = last_epoch_data.get("val_cm_tn")
            fp = last_epoch_data.get("val_cm_fp")
            fn = last_epoch_data.get("val_cm_fn")

            # Include both raw CM values and pre-computed stats
            if all(v is not None for v in [tp, tn, fp, fn]):
                confusion_matrix_obj = {
                    "true_positives": int(tp),
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    **persisted_stats,  # Add pre-computed stats
                }
            else:
                confusion_matrix_obj = persisted_stats.copy()
        else:
            # If no last epoch data, just use persisted stats
            confusion_matrix_obj = persisted_stats.copy()
    elif last_epoch_data:
        tp = last_epoch_data.get("val_cm_tp")
        tn = last_epoch_data.get("val_cm_tn")
        fp = last_epoch_data.get("val_cm_fp")
        fn = last_epoch_data.get("val_cm_fn")

        # Only build CM object if all values exist and are not None
        if all(v is not None for v in [tp, tn, fp, fn]):
            try:
                # Type guards: ensure values are not None before int() conversion
                if (
                    tp is not None
                    and tn is not None
                    and fp is not None
                    and fn is not None
                ):
                    cm_dict = {
                        "true_positives": int(tp),
                        "true_negatives": int(tn),
                        "false_positives": int(fp),
                        "false_negatives": int(fn),
                    }
                    # Calculate summary statistics
                    summary_stats = calculate_summary_statistics(cm_dict)
                    confusion_matrix_obj = {**cm_dict, **summary_stats}
                else:
                    confusion_matrix_obj = None
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
            "best_epoch": find_best_epoch(training_history),
            "best_val_accuracy": max(
                [(h.get("val_acc") or 0.0) for h in training_history],
                default=0.0,
            ),
            "best_val_precision": max(
                [(h.get("val_precision") or 0.0) for h in training_history],
                default=0.0,
            ),
            "best_val_recall": max(
                [(h.get("val_recall") or 0.0) for h in training_history],
                default=0.0,
            ),
            "best_val_loss": min(
                [(h.get("val_loss") or 0.0) for h in training_history],
                default=0.0,
            ),
            "best_val_f1": max(
                [(h.get("val_f1") or 0.0) for h in training_history],
                default=0.0,
            ),
            "best_val_auroc": max(
                [(h.get("val_auroc") or 0.0) for h in training_history],
                default=0.0,
            ),
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


def find_best_epoch(training_history: List[Dict]) -> int:
    """Find epoch with best validation accuracy.

    Note: Expects training_history with 1-indexed epochs (already transformed).
    Returns 1 as minimum since epochs are now displayed as 1-10 instead of 0-9.

    Args:
        training_history: List of epoch dictionaries with validation metrics

    Returns:
        Epoch number (1-indexed) with best validation accuracy
    """
    if not training_history:
        return 1  # Return 1 instead of 0 since epochs are now 1-indexed

    best_epoch = 1
    best_acc = 0.0

    for entry in training_history:
        val_acc = entry.get("val_acc") or 0.0
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = entry["epoch"]  # This is already 1-indexed from transformation

    return best_epoch
