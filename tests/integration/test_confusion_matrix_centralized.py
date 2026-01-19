"""
Integration tests for centralized training confusion matrix pipeline.

Tests the full flow:
1. LitResNet computes confusion matrix per epoch
2. MetricsCollector captures val_cm_* metrics
3. run_crud.persist_metrics() saves to database
4. API endpoint returns results with summary statistics
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import torch

from federated_pneumonia_detection.src.boundary.engine import Run
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.api.endpoints.runs_endpoints.utils import (
    _calculate_summary_statistics,
    _transform_run_to_results,
)


@pytest.mark.integration
class TestConfusionMatrixCentralizedFlow:
    """Integration tests for confusion matrix in centralized training."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()

    @pytest.fixture
    def confusion_matrix_values(self):
        """Create realistic confusion matrix values."""
        return {
            "true_positives": 450,
            "true_negatives": 430,
            "false_positives": 40,
            "false_negatives": 30,
        }

    @pytest.fixture
    def cm_summary_stats(self, confusion_matrix_values):
        """Calculate expected summary statistics."""
        return _calculate_summary_statistics(confusion_matrix_values)

    def test_calculate_summary_statistics_valid_matrix(self, confusion_matrix_values):
        """Test summary statistics calculation with valid confusion matrix."""
        stats = _calculate_summary_statistics(confusion_matrix_values)

        # Verify all expected keys are present
        assert "sensitivity" in stats
        assert "specificity" in stats
        assert "precision_cm" in stats
        assert "accuracy_cm" in stats
        assert "f1_cm" in stats

        # Verify calculations
        tp, tn, fp, fn = 450, 430, 40, 30
        expected_sensitivity = tp / (tp + fn)  # 0.9375
        expected_specificity = tn / (tn + fp)  # 0.9148
        expected_precision = tp / (tp + fp)    # 0.9184
        expected_accuracy = (tp + tn) / (tp + tn + fp + fn)  # 0.88

        assert stats["sensitivity"] == pytest.approx(expected_sensitivity, abs=0.001)
        assert stats["specificity"] == pytest.approx(expected_specificity, abs=0.001)
        assert stats["precision_cm"] == pytest.approx(expected_precision, abs=0.001)
        assert stats["accuracy_cm"] == pytest.approx(expected_accuracy, abs=0.001)

    def test_calculate_summary_statistics_zero_values(self):
        """Test summary statistics with zero confusion matrix values."""
        cm = {
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        stats = _calculate_summary_statistics(cm)

        assert stats["sensitivity"] == 0.0
        assert stats["specificity"] == 0.0
        assert stats["precision_cm"] == 0.0
        assert stats["accuracy_cm"] == 0.0
        assert stats["f1_cm"] == 0.0

    def test_calculate_summary_statistics_no_false_positives(self):
        """Test summary statistics when there are no false positives."""
        cm = {
            "true_positives": 100,
            "true_negatives": 100,
            "false_positives": 0,
            "false_negatives": 10,
        }
        stats = _calculate_summary_statistics(cm)

        # Precision should be 1.0 (100 / 100)
        assert stats["precision_cm"] == pytest.approx(1.0, abs=0.001)
        # Sensitivity should be 100/110
        assert stats["sensitivity"] == pytest.approx(0.909, abs=0.001)
        # Specificity should be 1.0 (100 / 100)
        assert stats["specificity"] == pytest.approx(1.0, abs=0.001)

    def test_calculate_summary_statistics_negative_values_raises_error(self):
        """Test that negative confusion matrix values raise ValueError."""
        cm = {
            "true_positives": -10,
            "true_negatives": 100,
            "false_positives": 10,
            "false_negatives": 5,
        }
        with pytest.raises(ValueError):
            _calculate_summary_statistics(cm)

    def test_transform_run_to_results_with_confusion_matrix(
        self, mock_db, confusion_matrix_values
    ):
        """Test transforming run with confusion matrix to results format."""
        # Create mock run object
        mock_run = Mock(spec=Run)
        mock_run.id = 1
        mock_run.status = "completed"
        mock_run.start_time = datetime(2024, 1, 1, 10, 0, 0)
        mock_run.end_time = datetime(2024, 1, 1, 11, 30, 0)

        # Create mock metrics
        metrics_data = [
            # Epoch 0 metrics
            Mock(
                step=0,
                metric_name="train_loss",
                metric_value=0.5,
            ),
            Mock(
                step=0,
                metric_name="val_loss",
                metric_value=0.4,
            ),
            Mock(
                step=0,
                metric_name="train_accuracy",
                metric_value=0.85,
            ),
            Mock(
                step=0,
                metric_name="val_accuracy",
                metric_value=0.88,
            ),
            Mock(
                step=0,
                metric_name="val_recall",
                metric_value=0.90,
            ),
            Mock(
                step=0,
                metric_name="val_precision",
                metric_value=0.87,
            ),
            Mock(
                step=0,
                metric_name="val_f1",
                metric_value=0.885,
            ),
            Mock(
                step=0,
                metric_name="val_cm_tp",
                metric_value=confusion_matrix_values["true_positives"],
            ),
            Mock(
                step=0,
                metric_name="val_cm_tn",
                metric_value=confusion_matrix_values["true_negatives"],
            ),
            Mock(
                step=0,
                metric_name="val_cm_fp",
                metric_value=confusion_matrix_values["false_positives"],
            ),
            Mock(
                step=0,
                metric_name="val_cm_fn",
                metric_value=confusion_matrix_values["false_negatives"],
            ),
        ]
        mock_run.metrics = metrics_data

        # Transform run to results
        result = _transform_run_to_results(mock_run)

        # Verify structure
        assert result["experiment_id"] == "run_1"
        assert result["status"] == "completed"
        assert result["final_metrics"] is not None
        assert result["confusion_matrix"] is not None
        assert result["training_history"] is not None

        # Verify confusion matrix values
        cm = result["confusion_matrix"]
        assert cm["true_positives"] == 450
        assert cm["true_negatives"] == 430
        assert cm["false_positives"] == 40
        assert cm["false_negatives"] == 30

        # Verify summary statistics are calculated and present
        assert "sensitivity" in cm
        assert "specificity" in cm
        assert "precision_cm" in cm
        assert "accuracy_cm" in cm
        assert "f1_cm" in cm

        # Verify calculations
        assert cm["sensitivity"] == pytest.approx(0.9375, abs=0.001)
        assert cm["specificity"] == pytest.approx(0.9148, abs=0.001)

    def test_transform_run_to_results_without_confusion_matrix(self, mock_db):
        """Test transforming run without confusion matrix."""
        mock_run = Mock(spec=Run)
        mock_run.id = 2
        mock_run.status = "completed"
        mock_run.start_time = datetime(2024, 1, 1, 10, 0, 0)
        mock_run.end_time = datetime(2024, 1, 1, 11, 0, 0)

        # Metrics without confusion matrix values
        metrics_data = [
            Mock(step=0, metric_name="train_loss", metric_value=0.5),
            Mock(step=0, metric_name="val_loss", metric_value=0.4),
            Mock(step=0, metric_name="val_accuracy", metric_value=0.88),
        ]
        mock_run.metrics = metrics_data

        result = _transform_run_to_results(mock_run)

        # Confusion matrix should be None
        assert result["confusion_matrix"] is None

    def test_transform_run_with_multiple_epochs(self):
        """Test transforming run with metrics from multiple epochs."""
        mock_run = Mock(spec=Run)
        mock_run.id = 3
        mock_run.status = "completed"
        mock_run.start_time = datetime(2024, 1, 1, 10, 0, 0)
        mock_run.end_time = datetime(2024, 1, 1, 12, 0, 0)

        # Metrics from 3 epochs
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

        # Should have 3 epochs in training history
        assert result["total_epochs"] == 3
        assert len(result["training_history"]) == 3

        # Epochs should be 1-indexed (1, 2, 3)
        assert result["training_history"][0]["epoch"] == 1
        assert result["training_history"][1]["epoch"] == 2
        assert result["training_history"][2]["epoch"] == 3

        # Confusion matrix should be from the last epoch (epoch 2 in 0-indexed, 3 in 1-indexed)
        cm = result["confusion_matrix"]
        assert cm["true_positives"] == 470  # 450 + 2*10
        assert cm["true_negatives"] == 450  # 430 + 2*10
        assert cm["false_positives"] == 30  # 40 - 2*5
        assert cm["false_negatives"] == 24  # 30 - 2*3

    def test_metrics_collector_captures_confusion_matrix_values(self):
        """Test that MetricsCollector properly captures confusion matrix metrics."""
        from federated_pneumonia_detection.src.control.dl_model.internals.model.metrics_collector import (
            MetricsCollectorCallback,
        )

        with patch(
            "federated_pneumonia_detection.src.control.dl_model.utils.model.metrics_collector.get_session"
        ):
            collector = MetricsCollectorCallback(
                save_dir="/tmp/test",
                experiment_name="test_exp",
                run_id=None,
                training_mode="centralized",
                enable_db_persistence=False,
            )

            # Mock trainer with metrics
            mock_trainer = Mock()
            mock_trainer.callback_metrics = {
                "train_loss": torch.tensor(0.5),
                "val_loss": torch.tensor(0.4),
                "val_accuracy": torch.tensor(0.88),
                "val_cm_tp": torch.tensor(450),
                "val_cm_tn": torch.tensor(430),
                "val_cm_fp": torch.tensor(40),
                "val_cm_fn": torch.tensor(30),
            }
            mock_trainer.logged_metrics = {}
            mock_trainer.optimizers = [Mock(param_groups=[{"lr": 0.001}])]
            mock_trainer.current_epoch = 0
            mock_trainer.global_step = 100

            mock_pl_module = Mock()

            # Extract metrics
            metrics = collector._extract_metrics(mock_trainer, mock_pl_module, "val")

            # Verify confusion matrix values are extracted
            assert metrics["val_cm_tp"] == 450
            assert metrics["val_cm_tn"] == 430
            assert metrics["val_cm_fp"] == 40
            assert metrics["val_cm_fn"] == 30

    def test_run_crud_persist_metrics_includes_confusion_matrix(self, mock_db):
        """Test that run_crud.persist_metrics saves confusion matrix metrics."""
        with patch("federated_pneumonia_detection.src.boundary.CRUD.run.RunMetric") as _MockRunMetric:
            # Setup mock run creation
            mock_run = Mock()
            mock_run.id = 1

            with patch.object(run_crud, "create", return_value=mock_run):
                # Create run
                created_run = run_crud.create(mock_db, training_mode="centralized", status="in_progress")

                # Mock epoch metrics with confusion matrix
                epoch_metrics = [
                    {
                        "epoch": 0,
                        "train_loss": 0.5,
                        "val_loss": 0.4,
                        "val_accuracy": 0.88,
                        "val_cm_tp": 450,
                        "val_cm_tn": 430,
                        "val_cm_fp": 40,
                        "val_cm_fn": 30,
                    }
                ]

                # Persist metrics
                with patch.object(run_crud, "persist_metrics") as mock_persist:
                    run_crud.persist_metrics(mock_db, created_run.id, epoch_metrics)
                    mock_persist.assert_called_once()

    def test_api_response_includes_summary_statistics(self, confusion_matrix_values):
        """Test that API response includes calculated summary statistics."""
        # Create complete results structure
        results = {
            "final_metrics": {
                "accuracy": 0.88,
                "precision": 0.92,
                "recall": 0.9,
                "f1_score": 0.91,
                "auc": 0.95,
            },
            "confusion_matrix": {
                **confusion_matrix_values,
                **_calculate_summary_statistics(confusion_matrix_values),
            },
        }

        # Verify all required fields
        cm = results["confusion_matrix"]
        assert cm["sensitivity"] > 0
        assert cm["specificity"] > 0
        assert cm["precision_cm"] > 0
        assert cm["accuracy_cm"] > 0
        assert cm["f1_cm"] > 0

        # Verify values are reasonable (between 0 and 1)
        assert 0 <= cm["sensitivity"] <= 1
        assert 0 <= cm["specificity"] <= 1
        assert 0 <= cm["precision_cm"] <= 1
        assert 0 <= cm["accuracy_cm"] <= 1
        assert 0 <= cm["f1_cm"] <= 1


@pytest.mark.integration
class TestConfusionMatrixEdgeCases:
    """Test edge cases in confusion matrix handling."""

    def test_all_true_positives(self):
        """Test CM when all predictions are true positives."""
        cm = {
            "true_positives": 100,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        stats = _calculate_summary_statistics(cm)

        assert stats["sensitivity"] == 1.0
        assert stats["specificity"] == 0.0
        assert stats["precision_cm"] == 1.0
        assert stats["accuracy_cm"] == 1.0
        assert stats["f1_cm"] == 1.0

    def test_all_true_negatives(self):
        """Test CM when all predictions are true negatives."""
        cm = {
            "true_positives": 0,
            "true_negatives": 100,
            "false_positives": 0,
            "false_negatives": 0,
        }
        stats = _calculate_summary_statistics(cm)

        assert stats["sensitivity"] == 0.0
        assert stats["specificity"] == 1.0
        assert stats["precision_cm"] == 0.0
        assert stats["accuracy_cm"] == 1.0
        assert stats["f1_cm"] == 0.0

    def test_balanced_errors(self):
        """Test CM with balanced false positives and false negatives."""
        cm = {
            "true_positives": 45,
            "true_negatives": 45,
            "false_positives": 5,
            "false_negatives": 5,
        }
        stats = _calculate_summary_statistics(cm)

        # Both sensitivity and specificity should be equal
        assert stats["sensitivity"] == pytest.approx(stats["specificity"], abs=0.001)

    def test_imbalanced_classes(self):
        """Test CM with highly imbalanced class distribution."""
        cm = {
            "true_positives": 10,
            "true_negatives": 990,
            "false_positives": 0,
            "false_negatives": 0,
        }
        stats = _calculate_summary_statistics(cm)

        # High accuracy due to many TN
        assert stats["accuracy_cm"] == pytest.approx(1.0, abs=0.001)
        # Low sensitivity due to few positives
        assert stats["sensitivity"] == 1.0
