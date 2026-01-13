"""
Standalone test file for confusion matrix functions.
Can run without conftest.py dependencies.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock
from federated_pneumonia_detection.src.api.endpoints.runs_endpoints.utils import (
    _calculate_summary_statistics,
    _transform_run_to_results,
)


def test_calculate_summary_statistics():
    """Test summary statistics calculation."""
    cm = {
        "true_positives": 450,
        "true_negatives": 430,
        "false_positives": 40,
        "false_negatives": 30,
    }

    stats = _calculate_summary_statistics(cm)

    assert "sensitivity" in stats
    assert "specificity" in stats
    assert "precision_cm" in stats
    assert "accuracy_cm" in stats
    assert "f1_cm" in stats

    # Verify calculations
    assert stats["sensitivity"] == pytest.approx(0.9375, abs=0.001)
    assert stats["specificity"] == pytest.approx(0.9148, abs=0.001)
    print("✓ Summary statistics calculation passed")


def test_calculate_summary_statistics_zero():
    """Test summary statistics with zero values."""
    cm = {
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }

    stats = _calculate_summary_statistics(cm)

    assert all(v == 0.0 for v in stats.values())
    print("✓ Zero confusion matrix test passed")


def test_calculate_summary_statistics_perfect():
    """Test summary statistics with perfect predictions."""
    cm = {
        "true_positives": 100,
        "true_negatives": 100,
        "false_positives": 0,
        "false_negatives": 0,
    }

    stats = _calculate_summary_statistics(cm)

    assert all(abs(v - 1.0) < 0.001 for v in stats.values())
    print("✓ Perfect confusion matrix test passed")


def test_transform_run_with_confusion_matrix():
    """Test transforming run with confusion matrix."""
    mock_run = Mock()
    mock_run.id = 1
    mock_run.status = "completed"
    mock_run.start_time = datetime(2024, 1, 1, 10, 0, 0)
    mock_run.end_time = datetime(2024, 1, 1, 11, 0, 0)

    metrics_data = [
        Mock(step=0, metric_name="train_loss", metric_value=0.5),
        Mock(step=0, metric_name="val_loss", metric_value=0.4),
        Mock(step=0, metric_name="val_accuracy", metric_value=0.88),
        Mock(step=0, metric_name="val_cm_tp", metric_value=450),
        Mock(step=0, metric_name="val_cm_tn", metric_value=430),
        Mock(step=0, metric_name="val_cm_fp", metric_value=40),
        Mock(step=0, metric_name="val_cm_fn", metric_value=30),
    ]
    mock_run.metrics = metrics_data

    result = _transform_run_to_results(mock_run)

    assert result["experiment_id"] == "run_1"
    assert result["status"] == "completed"
    assert result["confusion_matrix"] is not None

    cm = result["confusion_matrix"]
    assert cm["true_positives"] == 450
    assert cm["sensitivity"] > 0
    print("✓ Transform run with confusion matrix test passed")


def test_transform_run_without_confusion_matrix():
    """Test transforming run without confusion matrix."""
    mock_run = Mock()
    mock_run.id = 2
    mock_run.status = "completed"
    mock_run.start_time = datetime(2024, 1, 1, 10, 0, 0)
    mock_run.end_time = datetime(2024, 1, 1, 11, 0, 0)

    metrics_data = [
        Mock(step=0, metric_name="train_loss", metric_value=0.5),
        Mock(step=0, metric_name="val_loss", metric_value=0.4),
    ]
    mock_run.metrics = metrics_data

    result = _transform_run_to_results(mock_run)

    assert result["confusion_matrix"] is None
    print("✓ Transform run without confusion matrix test passed")


def test_multiple_epochs():
    """Test transforming run with multiple epochs."""
    mock_run = Mock()
    mock_run.id = 3
    mock_run.status = "completed"
    mock_run.start_time = datetime(2024, 1, 1, 10, 0, 0)
    mock_run.end_time = datetime(2024, 1, 1, 12, 0, 0)

    metrics_data = []
    for epoch in range(3):
        metrics_data.extend(
            [
                Mock(step=epoch, metric_name="train_loss", metric_value=0.5 - epoch * 0.1),
                Mock(step=epoch, metric_name="val_loss", metric_value=0.4 - epoch * 0.05),
                Mock(step=epoch, metric_name="val_accuracy", metric_value=0.80 + epoch * 0.03),
                Mock(step=epoch, metric_name="val_cm_tp", metric_value=450 + epoch * 10),
                Mock(step=epoch, metric_name="val_cm_tn", metric_value=430 + epoch * 10),
                Mock(step=epoch, metric_name="val_cm_fp", metric_value=40 - epoch * 5),
                Mock(step=epoch, metric_name="val_cm_fn", metric_value=30 - epoch * 3),
            ]
        )
    mock_run.metrics = metrics_data

    result = _transform_run_to_results(mock_run)

    assert result["total_epochs"] == 3
    assert len(result["training_history"]) == 3

    # Epochs should be 1-indexed
    assert result["training_history"][0]["epoch"] == 1
    assert result["training_history"][1]["epoch"] == 2
    assert result["training_history"][2]["epoch"] == 3

    # CM from last epoch
    cm = result["confusion_matrix"]
    assert cm["true_positives"] == 470  # 450 + 2*10
    print("✓ Multiple epochs test passed")


if __name__ == "__main__":
    print("\n=== Running Confusion Matrix Integration Tests ===\n")

    try:
        test_calculate_summary_statistics()
    except Exception as e:
        print(f"✗ Summary statistics test failed: {e}")

    test_calculate_summary_statistics_zero()
    test_calculate_summary_statistics_perfect()
    test_transform_run_with_confusion_matrix()
    test_transform_run_without_confusion_matrix()
    test_multiple_epochs()

    print("\n=== All Tests Passed ===\n")
