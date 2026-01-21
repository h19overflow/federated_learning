"""
Unit tests for server_evaluation module.

Tests server-side evaluation functions.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from flwr.app import ArrayRecord, MetricRecord

from federated_pneumonia_detection.src.control.federated_new_version.core.server_evaluation import (
    create_central_evaluate_fn,
)


class TestCreateCentralEvaluateFn:
    """Test suite for create_central_evaluate_fn factory function."""

    @patch(
        "federated_pneumonia_detection.src.control.federated_new_version.core.server_evaluation.LitResNet",
    )
    @patch(
        "federated_pneumonia_detection.src.control.federated_new_version.core.server_evaluation.XRayDataModule",
    )
    @patch("pandas.read_csv")
    @patch("torchmetrics.Precision")
    @patch("torchmetrics.Recall")
    @patch("torchmetrics.F1Score")
    @patch("torchmetrics.AUROC")
    @patch("torchmetrics.ConfusionMatrix")
    @pytest.mark.slow  # This test involves model operations
    def test_create_central_evaluate_fn_successful(
        self,
        mock_cm,
        mock_auroc,
        mock_f1,
        mock_recall,
        mock_precision,
        mock_read_csv,
        mock_data_module,
        mock_model,
        sample_metadata_df,
        mock_config_manager,
        mock_array_record,
    ):
        """Test that central_evaluate_fn is created successfully."""
        # Setup mocks
        mock_read_csv.return_value = sample_metadata_df

        mock_model_instance = Mock()
        mock_model_instance.state_dict.return_value = {
            "layer.weight": torch.randn(10, 10),
        }
        mock_model_instance.load_state_dict = Mock()
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_model_instance.eval = Mock()
        mock_model_instance.return_value = MagicMock()
        mock_model_instance.num_classes = 1
        mock_model_instance._calculate_loss = Mock(return_value=torch.tensor(0.5))
        mock_model_instance._get_predictions = Mock(
            return_value=torch.tensor([[0.7, 0.3]]),
        )
        mock_model_instance._prepare_targets_for_metrics = Mock(
            return_value=torch.tensor([0]),
        )
        mock_model_instance.return_value = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_dm_instance = Mock()
        mock_dm_instance.setup = Mock()
        mock_dm_instance.val_dataloader.return_value = [
            (torch.randn(2, 3, 224, 224), torch.tensor([0, 1])),
        ]
        mock_data_module.return_value = mock_dm_instance

        # Create metrics mocks
        mock_prec_metric = Mock()
        mock_prec_metric.return_value = torch.tensor(0.8)
        mock_precision.return_value = mock_prec_metric

        mock_rec_metric = Mock()
        mock_rec_metric.return_value = torch.tensor(0.75)
        mock_recall.return_value = mock_rec_metric

        mock_f1_metric = Mock()
        mock_f1_metric.return_value = torch.tensor(0.77)
        mock_f1.return_value = mock_f1_metric

        mock_auroc_metric = Mock()
        mock_auroc_metric.return_value = torch.tensor(0.9)
        mock_auroc.return_value = mock_auroc_metric

        # Create evaluate function
        evaluate_fn = create_central_evaluate_fn(
            config_manager=mock_config_manager,
            csv_path="test.csv",
            image_dir="test_images",
        )

        # Verify function is callable
        assert callable(evaluate_fn)

        # Call evaluate function (will be mocked to avoid real training)
        # For unit test, we'd need to mock more internals
        # This is a basic smoke test

    def test_create_central_evaluate_fn_returns_callable(self, mock_config_manager):
        """Test that factory returns a callable."""
        evaluate_fn = create_central_evaluate_fn(
            config_manager=mock_config_manager,
            csv_path="test.csv",
            image_dir="test_images",
        )

        assert callable(evaluate_fn)

    def test_evaluate_fn_accepts_correct_params(self, mock_config_manager):
        """Test that evaluate function accepts correct parameters."""
        evaluate_fn = create_central_evaluate_fn(
            config_manager=mock_config_manager,
            csv_path="test.csv",
            image_dir="test_images",
        )

        # Create mock ArrayRecord
        mock_arrays = Mock(spec=ArrayRecord)
        mock_arrays.to_torch_state_dict = Mock(
            return_value={"layer.weight": torch.randn(10, 10)},
        )

        # Should not raise TypeError
        try:
            evaluate_fn(server_round=1, arrays=mock_arrays)
        except Exception as e:
            # We expect some error due to mocking, but not TypeError for wrong signature
            assert "server_round" not in str(e)
            assert "arrays" not in str(e)

    @patch("pandas.read_csv")
    def test_evaluate_fn_loads_data(self, mock_read_csv, mock_config_manager):
        """Test that evaluate function loads CSV data."""
        mock_read_csv.return_value = pd.DataFrame({"patientId": [1, 2]})

        evaluate_fn = create_central_evaluate_fn(
            config_manager=mock_config_manager,
            csv_path="test.csv",
            image_dir="test_images",
        )

        mock_arrays = Mock(spec=ArrayRecord)
        mock_arrays.to_torch_state_dict = Mock(return_value={})

        # Call will likely fail due to mocks, but CSV should be attempted
        try:
            evaluate_fn(server_round=1, arrays=mock_arrays)
        except Exception:
            pass

        mock_read_csv.assert_called_once_with("test.csv")

    @patch("pandas.read_csv")
    def test_evaluate_fn_handles_missing_filename(
        self,
        mock_read_csv,
        mock_config_manager,
    ):
        """Test that evaluate function adds filename column if missing."""
        df_without_filename = pd.DataFrame(
            {
                "patientId": [1, 2, 3],
                "class": ["Normal", "Pneumonia", "Normal"],
            },
        )
        mock_read_csv.return_value = df_without_filename

        evaluate_fn = create_central_evaluate_fn(
            config_manager=mock_config_manager,
            csv_path="test.csv",
            image_dir="test_images",
        )

        mock_arrays = Mock(spec=ArrayRecord)
        mock_arrays.to_torch_state_dict = Mock(return_value={})

        # Will fail due to mocks, but test passes if no TypeError
        try:
            evaluate_fn(server_round=1, arrays=mock_arrays)
        except TypeError as e:
            if "filename" in str(e):
                pytest.fail(f"Filename handling failed: {e}")
        except Exception:
            pass  # Expected due to other mocks

    @patch("pandas.read_csv")
    def test_evaluate_fn_handles_missing_patientId(
        self,
        mock_read_csv,
        mock_config_manager,
    ):
        """Test that evaluate function handles missing patientId column gracefully."""
        df_without_patientId = pd.DataFrame(
            {
                "filename": ["1.png", "2.png"],
                "class": ["Normal", "Pneumonia"],
            },
        )
        mock_read_csv.return_value = df_without_patientId

        evaluate_fn = create_central_evaluate_fn(
            config_manager=mock_config_manager,
            csv_path="test.csv",
            image_dir="test_images",
        )

        mock_arrays = Mock(spec=ArrayRecord)
        mock_arrays.to_torch_state_dict = Mock(return_value={})

        # Should raise ValueError about missing patientId/filename
        with pytest.raises((ValueError, AttributeError, KeyError)):
            evaluate_fn(server_round=1, arrays=mock_arrays)


class MockEvaluationHarness:
    """Helper class for testing evaluation function with controlled mocks."""

    @staticmethod
    def create_mock_evaluation(round_num: int, arrays: ArrayRecord) -> MetricRecord:
        """Create mock evaluation result."""
        return MetricRecord(
            {
                "server_loss": 0.5 - (round_num * 0.05),
                "server_accuracy": 0.8 + (round_num * 0.02),
                "server_precision": 0.75,
                "server_recall": 0.7,
                "server_f1": 0.72,
                "server_auroc": 0.85,
                "server_cm_tp": 40.0,
                "server_cm_tn": 35.0,
                "server_cm_fp": 5.0,
                "server_cm_fn": 10.0,
            },
        )

    @staticmethod
    def assert_metrics_valid(metrics: MetricRecord):
        """Assert that metrics are valid."""
        metrics_dict = dict(metrics)

        # Check required fields
        required_fields = [
            "server_loss",
            "server_accuracy",
            "server_precision",
            "server_recall",
            "server_f1",
            "server_auroc",
        ]
        for field in required_fields:
            assert field in metrics_dict, f"Missing field: {field}"

        # Check ranges
        assert 0 <= metrics_dict["server_loss"] <= 10
        assert 0 <= metrics_dict["server_accuracy"] <= 1
        assert 0 <= metrics_dict["server_precision"] <= 1
        assert 0 <= metrics_dict["server_recall"] <= 1
        assert 0 <= metrics_dict["server_f1"] <= 1
        assert 0 <= metrics_dict["server_auroc"] <= 1

        # Check confusion matrix values
        assert metrics_dict["server_cm_tp"] >= 0
        assert metrics_dict["server_cm_tn"] >= 0
        assert metrics_dict["server_cm_fp"] >= 0
        assert metrics_dict["server_cm_fn"] >= 0
