from datetime import datetime
from unittest.mock import MagicMock

import pytest

from federated_pneumonia_detection.src.control.analytics.internals.utils import (
    calculate_summary_statistics,
    find_best_epoch,
    transform_run_to_results,
)

# --- Tests for calculate_summary_statistics ---


def test_calculate_summary_statistics_happy_path():
    cm = {
        "true_positives": 80,
        "true_negatives": 90,
        "false_positives": 10,
        "false_negatives": 20,
    }
    stats = calculate_summary_statistics(cm)

    # Sensitivity = 80 / (80 + 20) = 0.8
    assert stats["sensitivity"] == 0.8
    # Specificity = 90 / (90 + 10) = 0.9
    assert stats["specificity"] == 0.9
    # Precision = 80 / (80 + 10) = 0.8889
    assert stats["precision_cm"] == 0.8889
    # Accuracy = (80 + 90) / 200 = 0.85
    assert stats["accuracy_cm"] == 0.85
    # F1 = 2 * (0.8889 * 0.8) / (0.8889 + 0.8) = 0.8421
    assert stats["f1_cm"] == 0.8421


def test_calculate_summary_statistics_zero_total():
    cm = {
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    stats = calculate_summary_statistics(cm)
    assert all(v == 0.0 for v in stats.values())


def test_calculate_summary_statistics_negative_values():
    cm = {
        "true_positives": -1,
        "true_negatives": 90,
        "false_positives": 10,
        "false_negatives": 20,
    }
    with pytest.raises(ValueError, match="Confusion matrix values cannot be negative"):
        calculate_summary_statistics(cm)


def test_calculate_summary_statistics_zero_division_cases():
    # Case where precision denominator is zero
    cm = {
        "true_positives": 0,
        "true_negatives": 100,
        "false_positives": 0,
        "false_negatives": 50,
    }
    stats = calculate_summary_statistics(cm)
    assert stats["precision_cm"] == 0.0
    assert stats["sensitivity"] == 0.0
    assert stats["f1_cm"] == 0.0
    assert stats["specificity"] == 1.0


# --- Tests for find_best_epoch ---


def test_find_best_epoch_happy_path():
    history = [
        {"epoch": 1, "val_acc": 0.5},
        {"epoch": 2, "val_acc": 0.8},
        {"epoch": 3, "val_acc": 0.7},
    ]
    assert find_best_epoch(history) == 2


def test_find_best_epoch_empty():
    assert find_best_epoch([]) == 1


def test_find_best_epoch_tie():
    history = [{"epoch": 1, "val_acc": 0.8}, {"epoch": 2, "val_acc": 0.8}]
    # Should return the first one encountered
    assert find_best_epoch(history) == 1


# --- Tests for transform_run_to_results ---


def test_transform_run_to_results_centralized():
    mock_run = MagicMock()
    mock_run.id = 123
    mock_run.status = "completed"
    mock_run.training_mode = "centralized"
    mock_run.start_time = datetime(2023, 1, 1, 10, 0, 0)
    mock_run.end_time = datetime(2023, 1, 1, 11, 0, 0)

    # Mock metrics
    m1 = MagicMock(step=0, metric_name="val_acc", metric_value=0.8)
    m2 = MagicMock(step=0, metric_name="val_loss", metric_value=0.4)
    m3 = MagicMock(step=0, metric_name="val_cm_tp", metric_value=10)
    m4 = MagicMock(step=0, metric_name="val_cm_tn", metric_value=10)
    m5 = MagicMock(step=0, metric_name="val_cm_fp", metric_value=0)
    m6 = MagicMock(step=0, metric_name="val_cm_fn", metric_value=0)

    mock_run.metrics = [m1, m2, m3, m4, m5, m6]

    result = transform_run_to_results(mock_run)

    assert result["experiment_id"] == "run_123"
    assert result["status"] == "completed"
    assert result["final_metrics"]["accuracy"] == 0.8
    assert len(result["training_history"]) == 1
    assert result["training_history"][0]["epoch"] == 1  # 0-indexed + 1
    assert result["confusion_matrix"]["true_positives"] == 10
    assert result["metadata"]["best_epoch"] == 1


def test_transform_run_to_results_federated():
    mock_run = MagicMock()
    mock_run.id = 456
    mock_run.status = "completed"
    mock_run.training_mode = "federated"
    mock_run.start_time = datetime(2023, 1, 1, 10, 0, 0)
    mock_run.end_time = None

    # Mock server evaluations
    e1 = MagicMock(
        round_number=1,
        loss=0.5,
        accuracy=0.75,
        precision=0.7,
        recall=0.8,
        f1_score=0.75,
        auroc=0.85,
        true_positives=15,
        true_negatives=15,
        false_positives=5,
        false_negatives=5,
    )
    mock_run.server_evaluations = [e1]

    result = transform_run_to_results(mock_run)

    assert result["experiment_id"] == "run_456"
    assert result["final_metrics"]["accuracy"] == 0.75
    assert result["training_history"][0]["epoch"] == 1  # round_number as-is
    assert result["metadata"]["end_time"] == ""
    assert result["confusion_matrix"]["accuracy_cm"] == 0.75


def test_transform_run_to_results_with_persisted_stats():
    mock_run = MagicMock()
    mock_run.id = 789
    mock_run.training_mode = "centralized"
    mock_run.metrics = []
    mock_run.start_time = None
    mock_run.end_time = None

    persisted = {"accuracy_cm": 0.99, "f1_cm": 0.98}

    result = transform_run_to_results(mock_run, persisted_stats=persisted)

    assert result["confusion_matrix"] == persisted
    # Ensure it's a copy
    assert result["confusion_matrix"] is not persisted


def test_transform_run_to_results_missing_cm_values():
    mock_run = MagicMock()
    mock_run.id = 101
    mock_run.training_mode = "centralized"
    mock_run.start_time = None
    mock_run.end_time = None

    # Only some CM values
    m1 = MagicMock(step=0, metric_name="val_cm_tp", metric_value=10)
    mock_run.metrics = [m1]

    result = transform_run_to_results(mock_run)

    assert result["confusion_matrix"] is None
