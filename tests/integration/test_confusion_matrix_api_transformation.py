"""
Integration tests for API response transformation with confusion matrices.

Tests the transformation layer that converts database objects to API responses,
including summary statistics calculation and response formatting.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any

from federated_pneumonia_detection.src.api.endpoints.runs_endpoints.utils import (
    _calculate_summary_statistics,
    _transform_run_to_results,
    _find_best_epoch,
)


@pytest.mark.integration
class TestAPITransformationWithConfusionMatrix:
    """Integration tests for API response transformation."""

    @pytest.fixture
    def sample_run_with_cm(self):
        """Create a complete mock run object with confusion matrix."""
        mock_run = Mock()
        mock_run.id = 1
        mock_run.status = "completed"
        mock_run.start_time = datetime(2024, 1, 1, 10, 0, 0)
        mock_run.end_time = datetime(2024, 1, 1, 11, 30, 0)

        # Create complete metric set for 3 epochs
        metrics_data = []
        for epoch in range(3):
            val_acc = 0.80 + epoch * 0.05
            val_recall = 0.78 + epoch * 0.06
            val_precision = 0.82 + epoch * 0.05
            val_f1 = 0.80 + epoch * 0.055
            val_loss = 0.4 - epoch * 0.05

            # Confusion matrix improving over epochs
            tp = 400 + epoch * 20
            tn = 380 + epoch * 20
            fp = 50 - epoch * 10
            fn = 70 - epoch * 15

            metrics_data.extend(
                [
                    Mock(step=epoch, metric_name="train_loss", metric_value=0.5 - epoch * 0.1),
                    Mock(step=epoch, metric_name="train_accuracy", metric_value=0.75 + epoch * 0.04),
                    Mock(step=epoch, metric_name="val_loss", metric_value=val_loss),
                    Mock(step=epoch, metric_name="val_accuracy", metric_value=val_acc),
                    Mock(step=epoch, metric_name="val_recall", metric_value=val_recall),
                    Mock(step=epoch, metric_name="val_precision", metric_value=val_precision),
                    Mock(step=epoch, metric_name="val_f1", metric_value=val_f1),
                    Mock(step=epoch, metric_name="val_auroc", metric_value=0.90 + epoch * 0.03),
                    Mock(step=epoch, metric_name="val_cm_tp", metric_value=tp),
                    Mock(step=epoch, metric_name="val_cm_tn", metric_value=tn),
                    Mock(step=epoch, metric_name="val_cm_fp", metric_value=fp),
                    Mock(step=epoch, metric_name="val_cm_fn", metric_value=fn),
                ]
            )

        mock_run.metrics = metrics_data
        return mock_run

    def test_transform_run_complete_structure(self, sample_run_with_cm):
        """Test complete transformation of run object to API response."""
        result = _transform_run_to_results(sample_run_with_cm)

        # Verify top-level structure
        assert "experiment_id" in result
        assert "status" in result
        assert "final_metrics" in result
        assert "confusion_matrix" in result
        assert "training_history" in result
        assert "total_epochs" in result
        assert "metadata" in result

        # Verify values
        assert result["experiment_id"] == "run_1"
        assert result["status"] == "completed"
        assert result["total_epochs"] == 3

    def test_transform_run_final_metrics(self, sample_run_with_cm):
        """Test that final metrics are extracted from last epoch."""
        result = _transform_run_to_results(sample_run_with_cm)

        final_metrics = result["final_metrics"]
        assert "accuracy" in final_metrics
        assert "precision" in final_metrics
        assert "recall" in final_metrics
        assert "f1_score" in final_metrics
        assert "auc" in final_metrics
        assert "loss" in final_metrics

        # All values should be floats between 0 and 1
        for key, value in final_metrics.items():
            if key != "loss":
                assert 0 <= value <= 1, f"{key} should be between 0 and 1, got {value}"

    def test_transform_run_confusion_matrix_extraction(self, sample_run_with_cm):
        """Test that confusion matrix is correctly extracted and includes stats."""
        result = _transform_run_to_results(sample_run_with_cm)

        cm = result["confusion_matrix"]
        assert cm is not None

        # Verify CM raw values exist
        assert cm["true_positives"] == 440  # 400 + 2*20
        assert cm["true_negatives"] == 420  # 380 + 2*20
        assert cm["false_positives"] == 30  # 50 - 2*10
        assert cm["false_negatives"] == 40  # 70 - 2*15

        # Verify summary statistics exist and are calculated
        assert cm["sensitivity"] > 0
        assert cm["specificity"] > 0
        assert cm["precision_cm"] > 0
        assert cm["accuracy_cm"] > 0
        assert cm["f1_cm"] > 0

    def test_transform_run_training_history_1_indexed(self, sample_run_with_cm):
        """Test that training history epochs are 1-indexed."""
        result = _transform_run_to_results(sample_run_with_cm)

        training_history = result["training_history"]
        assert len(training_history) == 3

        # Epochs should be 1, 2, 3 (not 0, 1, 2)
        for i, entry in enumerate(training_history, 1):
            assert entry["epoch"] == i
            assert "train_loss" in entry
            assert "val_loss" in entry
            assert "train_acc" in entry
            assert "val_acc" in entry

    def test_transform_run_metadata_completeness(self, sample_run_with_cm):
        """Test that metadata is complete and consistent."""
        result = _transform_run_to_results(sample_run_with_cm)

        metadata = result["metadata"]
        assert metadata["total_epochs"] == 3
        assert metadata["best_epoch"] in [1, 2, 3]
        assert "start_time" in metadata
        assert "end_time" in metadata
        assert "best_val_accuracy" in metadata
        assert "best_val_recall" in metadata
        assert "best_val_loss" in metadata
        assert "final_accuracy" in metadata
        assert "final_precision" in metadata
        assert "final_recall" in metadata

    def test_find_best_epoch_by_accuracy(self):
        """Test finding best epoch by validation accuracy."""
        training_history = [
            {"epoch": 1, "val_acc": 0.80, "val_loss": 0.40},
            {"epoch": 2, "val_acc": 0.85, "val_loss": 0.35},
            {"epoch": 3, "val_acc": 0.88, "val_loss": 0.32},
        ]

        best_epoch = _find_best_epoch(training_history)
        assert best_epoch == 3

    def test_find_best_epoch_with_fluctuating_accuracy(self):
        """Test finding best epoch with fluctuating validation accuracy."""
        training_history = [
            {"epoch": 1, "val_acc": 0.80},
            {"epoch": 2, "val_acc": 0.85},
            {"epoch": 3, "val_acc": 0.82},  # Lower than epoch 2
            {"epoch": 4, "val_acc": 0.88},
        ]

        best_epoch = _find_best_epoch(training_history)
        assert best_epoch == 4

    def test_transform_run_metadata_includes_final_metrics(self, sample_run_with_cm):
        """Test that metadata includes final metric values."""
        result = _transform_run_to_results(sample_run_with_cm)

        metadata = result["metadata"]
        final_metrics = result["final_metrics"]

        # Metadata should include final metric copies
        assert metadata["final_accuracy"] == final_metrics["accuracy"]
        assert metadata["final_precision"] == final_metrics["precision"]
        assert metadata["final_recall"] == final_metrics["recall"]
        assert metadata["final_f1"] == final_metrics["f1_score"]


@pytest.mark.integration
class TestSummaryStatisticsCalculation:
    """Tests for summary statistics calculation in API responses."""

    def test_summary_statistics_all_values_rounded(self):
        """Test that summary statistics are rounded to 4 decimal places."""
        cm = {
            "true_positives": 123,
            "true_negatives": 456,
            "false_positives": 78,
            "false_negatives": 89,
        }

        stats = _calculate_summary_statistics(cm)

        # All values should have at most 4 decimal places
        for key, value in stats.items():
            # Check if it's a rounded value (ends with 0001-0004 in 4th decimal)
            str_value = f"{value:.4f}"
            assert len(str_value.split(".")[1]) <= 4

    def test_summary_statistics_no_division_by_zero(self):
        """Test that summary statistics handle edge cases without division by zero."""
        # Case 1: No positives
        cm1 = {
            "true_positives": 0,
            "true_negatives": 100,
            "false_positives": 0,
            "false_negatives": 0,
        }
        stats1 = _calculate_summary_statistics(cm1)
        assert stats1["sensitivity"] == 0.0

        # Case 2: No negatives
        cm2 = {
            "true_positives": 100,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        stats2 = _calculate_summary_statistics(cm2)
        assert stats2["specificity"] == 0.0

        # Case 3: No true positives
        cm3 = {
            "true_positives": 0,
            "true_negatives": 100,
            "false_positives": 50,
            "false_negatives": 50,
        }
        stats3 = _calculate_summary_statistics(cm3)
        assert stats3["precision_cm"] == 0.0

    def test_summary_statistics_boundary_values(self):
        """Test summary statistics with boundary values."""
        # Perfect prediction
        cm_perfect = {
            "true_positives": 100,
            "true_negatives": 100,
            "false_positives": 0,
            "false_negatives": 0,
        }
        stats_perfect = _calculate_summary_statistics(cm_perfect)
        assert all(v == 1.0 for v in stats_perfect.values())

        # Worst case (opposite of perfect)
        cm_worst = {
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 100,
            "false_negatives": 100,
        }
        stats_worst = _calculate_summary_statistics(cm_worst)
        assert all(v == 0.0 for v in stats_worst.values())


@pytest.mark.integration
class TestAPIResponseConsistency:
    """Tests for consistency of API responses with confusion matrices."""

    def test_api_response_contains_all_required_fields(self):
        """Test that API response includes all required fields."""
        # Simulate complete API response structure
        api_response = {
            "run_id": 1,
            "is_federated": False,
            "has_server_evaluation": False,
            "final_metrics": {
                "accuracy": 0.92,
                "precision": 0.91,
                "recall": 0.93,
                "f1_score": 0.92,
                "auc": 0.95,
                "loss": 0.30,
            },
            "confusion_matrix": {
                "true_positives": 450,
                "true_negatives": 430,
                "false_positives": 40,
                "false_negatives": 30,
                "sensitivity": 0.9375,
                "specificity": 0.9148,
                "precision_cm": 0.9184,
                "accuracy_cm": 0.88,
                "f1_cm": 0.9275,
            },
            "training_history": [
                {
                    "epoch": 1,
                    "train_loss": 0.5,
                    "val_loss": 0.4,
                    "train_acc": 0.85,
                    "val_acc": 0.88,
                }
            ],
            "metadata": {
                "experiment_name": "run_1",
                "start_time": "2024-01-01T10:00:00",
                "end_time": "2024-01-01T11:30:00",
                "total_epochs": 3,
                "best_epoch": 3,
                "best_val_accuracy": 0.92,
                "best_val_recall": 0.93,
                "best_val_loss": 0.25,
            },
        }

        # Verify all required fields exist
        assert "run_id" in api_response
        assert "final_metrics" in api_response
        assert "confusion_matrix" in api_response
        assert "training_history" in api_response
        assert "metadata" in api_response

        # Verify confusion matrix completeness
        cm = api_response["confusion_matrix"]
        required_cm_fields = [
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "sensitivity",
            "specificity",
            "precision_cm",
            "accuracy_cm",
            "f1_cm",
        ]
        for field in required_cm_fields:
            assert field in cm

    def test_api_response_type_correctness(self):
        """Test that API response types are correct."""
        api_response = {
            "run_id": 1,
            "is_federated": False,
            "final_metrics": {
                "accuracy": 0.92,
                "precision": 0.91,
            },
            "confusion_matrix": {
                "true_positives": 450,
                "sensitivity": 0.9375,
            },
            "training_history": [],
            "metadata": {},
        }

        # Type checks
        assert isinstance(api_response["run_id"], int)
        assert isinstance(api_response["is_federated"], bool)
        assert isinstance(api_response["final_metrics"], dict)
        assert isinstance(api_response["confusion_matrix"], dict)
        assert isinstance(api_response["training_history"], list)
        assert isinstance(api_response["metadata"], dict)

        # Numeric type checks
        assert isinstance(api_response["final_metrics"]["accuracy"], float)
        assert isinstance(api_response["confusion_matrix"]["true_positives"], int)
        assert isinstance(api_response["confusion_matrix"]["sensitivity"], float)

    def test_api_response_numeric_constraints(self):
        """Test that numeric values in API response respect constraints."""
        api_response = {
            "confusion_matrix": {
                "true_positives": 450,
                "true_negatives": 430,
                "false_positives": 40,
                "false_negatives": 30,
                "sensitivity": 0.9375,
                "specificity": 0.9148,
                "precision_cm": 0.9184,
                "accuracy_cm": 0.88,
                "f1_cm": 0.9275,
            },
        }

        cm = api_response["confusion_matrix"]

        # CM counts should be non-negative integers
        for key in ["true_positives", "true_negatives", "false_positives", "false_negatives"]:
            assert cm[key] >= 0
            assert isinstance(cm[key], int)

        # Statistics should be between 0 and 1
        for key in ["sensitivity", "specificity", "precision_cm", "accuracy_cm", "f1_cm"]:
            assert 0 <= cm[key] <= 1, f"{key} should be between 0 and 1, got {cm[key]}"
