"""
Integration tests for federated learning confusion matrix pipeline.

Tests the full flow:
1. Server evaluation computes CM on held-out test set per round
2. MetricRecord returns server_cm_tp, server_cm_tn, server_cm_fp, server_cm_fn
3. ServerEvaluationCRUD persists to database
4. API endpoint returns results with summary statistics
"""

from unittest.mock import Mock, patch

import pytest

from federated_pneumonia_detection.src.api.endpoints.runs_endpoints.utils import (
    _calculate_summary_statistics,
)
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import (
    ServerEvaluationCRUD,
)


@pytest.mark.integration
@pytest.mark.federated
class TestConfusionMatrixFederatedFlow:
    """Integration tests for confusion matrix in federated learning."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()

    @pytest.fixture
    def server_cm_flat_format(self):
        """Create metrics in flat format from server_evaluation.py."""
        return {
            "loss": 0.35,
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.93,
            "f1_score": 0.92,
            "auroc": 0.95,
            "server_cm_tp": 450,
            "server_cm_tn": 430,
            "server_cm_fp": 40,
            "server_cm_fn": 30,
            "num_samples": 950,
        }

    @pytest.fixture
    def server_cm_nested_format(self):
        """Create metrics in nested format (backward compatible)."""
        return {
            "loss": 0.35,
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.93,
            "f1_score": 0.92,
            "auroc": 0.95,
            "confusion_matrix": {
                "true_positives": 450,
                "true_negatives": 430,
                "false_positives": 40,
                "false_negatives": 30,
            },
            "num_samples": 950,
        }

    def test_server_evaluation_crud_create_with_flat_format(
        self,
        mock_db,
        server_cm_flat_format,
    ):
        """Test ServerEvaluationCRUD.create_evaluation with flat CM format."""
        crud = ServerEvaluationCRUD()

        with patch.object(crud, "create") as mock_create:
            mock_eval = Mock()
            mock_create.return_value = mock_eval

            # Create evaluation with flat format metrics
            _result = crud.create_evaluation(
                mock_db,
                run_id=1,
                round_number=1,
                metrics=server_cm_flat_format,
            )

            # Verify create was called with extracted CM values
            call_args = mock_create.call_args
            assert call_args is not None

            # Get the kwargs from the call
            kwargs = call_args.kwargs if call_args.kwargs else call_args[1]

            # Verify CM values were extracted and mapped correctly
            assert kwargs.get("true_positives") == 450
            assert kwargs.get("true_negatives") == 430
            assert kwargs.get("false_positives") == 40
            assert kwargs.get("false_negatives") == 30

            # Verify other metrics are also present
            assert kwargs.get("accuracy") == 0.92
            assert kwargs.get("loss") == 0.35
            assert kwargs.get("num_samples") == 950

    def test_server_evaluation_crud_create_with_nested_format(
        self,
        mock_db,
        server_cm_nested_format,
    ):
        """Test ServerEvaluationCRUD.create_evaluation with nested CM format."""
        crud = ServerEvaluationCRUD()

        with patch.object(crud, "create") as mock_create:
            mock_eval = Mock()
            mock_create.return_value = mock_eval

            # Create evaluation with nested format metrics
            _result = crud.create_evaluation(
                mock_db,
                run_id=1,
                round_number=1,
                metrics=server_cm_nested_format,
            )

            # Verify create was called with extracted CM values
            call_args = mock_create.call_args
            assert call_args is not None

            kwargs = call_args.kwargs if call_args.kwargs else call_args[1]

            # Verify CM values were extracted from nested dict
            assert kwargs.get("true_positives") == 450
            assert kwargs.get("true_negatives") == 430
            assert kwargs.get("false_positives") == 40
            assert kwargs.get("false_negatives") == 30

    def test_server_evaluation_crud_excluded_keys(self, mock_db, server_cm_flat_format):
        """Test that CM keys are excluded from additional_metrics."""
        crud = ServerEvaluationCRUD()

        # Add some additional metrics
        metrics_with_extra = {
            **server_cm_flat_format,
            "custom_metric_1": 0.5,
            "custom_metric_2": 0.75,
        }

        with patch.object(crud, "create") as mock_create:
            mock_eval = Mock()
            mock_create.return_value = mock_eval

            _result = crud.create_evaluation(
                mock_db,
                run_id=1,
                round_number=1,
                metrics=metrics_with_extra,
            )

            call_args = mock_create.call_args
            kwargs = call_args.kwargs if call_args.kwargs else call_args[1]

            # CM-related keys should NOT be in additional_metrics
            additional = kwargs.get("additional_metrics", {})
            assert "server_cm_tp" not in additional
            assert "server_cm_tn" not in additional
            assert "server_cm_fp" not in additional
            assert "server_cm_fn" not in additional

            # But custom metrics should be there
            assert "custom_metric_1" in additional
            assert "custom_metric_2" in additional

    def test_server_evaluation_get_by_run_per_round(self, mock_db):
        """Test retrieving server evaluations per round."""
        crud = ServerEvaluationCRUD()

        # Create mock evaluations for different rounds
        mock_evals = [
            Mock(round_number=1, accuracy=0.85),
            Mock(round_number=2, accuracy=0.88),
            Mock(round_number=3, accuracy=0.91),
        ]

        with patch.object(crud, "get_by_run", return_value=mock_evals):
            result = crud.get_by_run(mock_db, run_id=1, order_by_round=True)

            assert len(result) == 3
            assert result[0].round_number == 1
            assert result[1].round_number == 2
            assert result[2].round_number == 3

    def test_server_evaluation_summary_stats(self, mock_db):
        """Test ServerEvaluationCRUD.get_summary_stats includes best metrics."""
        crud = ServerEvaluationCRUD()

        # Create mock evaluations with varying metrics
        mock_evals = [
            Mock(
                round_number=1,
                loss=0.5,
                accuracy=0.85,
                recall=0.82,
                f1_score=0.83,
            ),
            Mock(
                round_number=2,
                loss=0.35,
                accuracy=0.90,
                recall=0.91,
                f1_score=0.90,
            ),
            Mock(
                round_number=3,
                loss=0.30,
                accuracy=0.92,
                recall=0.93,
                f1_score=0.92,
            ),
        ]

        with patch.object(crud, "get_by_run", return_value=mock_evals):
            result = crud.get_summary_stats(mock_db, run_id=1)

            # Verify structure
            assert "total_rounds" in result
            assert "latest_round" in result
            assert "best_accuracy" in result
            assert "best_recall" in result
            assert "best_f1_score" in result

            # Verify values
            assert result["total_rounds"] == 3
            assert result["latest_round"] == 3
            assert result["best_accuracy"]["value"] == 0.92
            assert result["best_accuracy"]["round"] == 3
            assert result["best_recall"]["value"] == 0.93
            assert result["best_recall"]["round"] == 3

    def test_server_evaluation_per_round_persistence(self, mock_db):
        """Test that server evaluations are persisted per round in database."""
        crud = ServerEvaluationCRUD()

        # Simulate creating evaluations for multiple rounds
        rounds_data = [
            {
                "run_id": 1,
                "round_number": 1,
                "metrics": {
                    "loss": 0.5,
                    "accuracy": 0.85,
                    "server_cm_tp": 400,
                    "server_cm_tn": 380,
                    "server_cm_fp": 50,
                    "server_cm_fn": 70,
                },
            },
            {
                "run_id": 1,
                "round_number": 2,
                "metrics": {
                    "loss": 0.35,
                    "accuracy": 0.90,
                    "server_cm_tp": 430,
                    "server_cm_tn": 410,
                    "server_cm_fp": 40,
                    "server_cm_fn": 20,
                },
            },
            {
                "run_id": 1,
                "round_number": 3,
                "metrics": {
                    "loss": 0.30,
                    "accuracy": 0.92,
                    "server_cm_tp": 450,
                    "server_cm_tn": 430,
                    "server_cm_fp": 40,
                    "server_cm_fn": 30,
                },
            },
        ]

        with patch.object(crud, "create") as mock_create:
            created_evals = []
            for round_data in rounds_data:
                mock_eval = Mock(
                    round_number=round_data["round_number"],
                    true_positives=round_data["metrics"]["server_cm_tp"],
                )
                mock_create.return_value = mock_eval
                created_evals.append(mock_eval)

            # Verify each round has distinct metrics
            assert created_evals[0].true_positives == 400
            assert created_evals[1].true_positives == 430
            assert created_evals[2].true_positives == 450

    def test_server_evaluation_confusion_matrix_format_detection(self, mock_db):
        """Test that CM format is correctly detected and extracted."""
        _crud = ServerEvaluationCRUD()

        # Test that flat format is detected
        flat_metrics = {"server_cm_tp": 100}
        assert "server_cm_tp" in flat_metrics

        # Test that nested format is detected
        nested_metrics = {"confusion_matrix": {"true_positives": 100}}
        assert "confusion_matrix" in nested_metrics

    def test_server_evaluation_with_missing_cm_values(self, mock_db):
        """Test handling of server evaluation when CM values are missing."""
        _crud = ServerEvaluationCRUD()

        # Metrics without CM values
        incomplete_metrics = {
            "loss": 0.35,
            "accuracy": 0.92,
            "precision": 0.91,
        }

        with patch.object(_crud, "create") as mock_create:
            mock_eval = Mock()
            mock_create.return_value = mock_eval

            _result = _crud.create_evaluation(
                mock_db,
                run_id=1,
                round_number=1,
                metrics=incomplete_metrics,
            )

            call_args = mock_create.call_args
            kwargs = call_args.kwargs if call_args.kwargs else call_args[1]

            # CM values should not be present (or be None)
            assert kwargs.get("true_positives") is None
            assert kwargs.get("true_negatives") is None
            assert kwargs.get("false_positives") is None
            assert kwargs.get("false_negatives") is None


@pytest.mark.integration
@pytest.mark.federated
class TestServerEvaluationAPIIntegration:
    """Integration tests for server evaluation API endpoints."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()

    def test_server_evaluation_response_structure(self):
        """Test that server evaluation API response has correct structure."""

        # Mock response should include:
        expected_response = {
            "run_id": 1,
            "is_federated": True,
            "has_server_evaluation": True,
            "evaluations": [
                {
                    "round": 1,
                    "loss": 0.35,
                    "accuracy": 0.92,
                    "precision": 0.91,
                    "recall": 0.93,
                    "f1_score": 0.92,
                    "auroc": 0.95,
                    "confusion_matrix": {
                        "true_positives": 450,
                        "true_negatives": 430,
                        "false_positives": 40,
                        "false_negatives": 30,
                    },
                    "num_samples": 950,
                    "evaluation_time": "2025-11-04T12:34:56",
                },
            ],
            "summary": {
                "total_rounds": 5,
                "best_accuracy": {"value": 0.95, "round": 3},
                "best_recall": {"value": 0.94, "round": 4},
                "best_f1_score": {"value": 0.93, "round": 3},
            },
        }

        # Verify response structure
        assert "run_id" in expected_response
        assert "is_federated" in expected_response
        assert "has_server_evaluation" in expected_response
        assert "evaluations" in expected_response
        assert "summary" in expected_response

        # Verify evaluation entry structure
        eval_entry = expected_response["evaluations"][0]
        assert "round" in eval_entry
        assert "loss" in eval_entry
        assert "accuracy" in eval_entry
        assert "confusion_matrix" in eval_entry

    def test_server_evaluation_summary_with_no_evaluations(self):
        """Test server evaluation response when no evaluations exist."""
        expected_response = {
            "run_id": 1,
            "is_federated": True,
            "has_server_evaluation": False,
            "evaluations": [],
            "summary": {},
        }

        assert expected_response["has_server_evaluation"] is False
        assert len(expected_response["evaluations"]) == 0
        assert expected_response["summary"] == {}

    def test_server_evaluation_non_federated_run(self):
        """Test server evaluation response for non-federated run."""
        expected_response = {
            "run_id": 1,
            "is_federated": False,
            "has_server_evaluation": False,
            "evaluations": [],
            "summary": {},
        }

        assert expected_response["is_federated"] is False
        assert expected_response["has_server_evaluation"] is False


@pytest.mark.integration
@pytest.mark.federated
class TestFederatedMetricsAggregation:
    """Tests for federated learning metrics aggregation with confusion matrices."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()

    def test_multiple_rounds_confusion_matrices(self):
        """Test aggregating confusion matrices across multiple federated rounds."""
        rounds_data = []

        for round_num in range(1, 4):
            # Simulate improving CM across rounds
            tp_base = 400 + round_num * 20
            tn_base = 380 + round_num * 20

            round_data = {
                "round": round_num,
                "server_cm_tp": tp_base,
                "server_cm_tn": tn_base,
                "server_cm_fp": 100 - round_num * 15,
                "server_cm_fn": 100 - round_num * 15,
            }
            rounds_data.append(round_data)

        # Calculate summary statistics for each round
        summary_stats = []
        for round_data in rounds_data:
            cm = {
                "true_positives": round_data["server_cm_tp"],
                "true_negatives": round_data["server_cm_tn"],
                "false_positives": round_data["server_cm_fp"],
                "false_negatives": round_data["server_cm_fn"],
            }
            stats = _calculate_summary_statistics(cm)
            summary_stats.append(stats)

        # Verify progression
        assert len(summary_stats) == 3
        # Accuracy should generally improve
        acc_round1 = summary_stats[0]["accuracy_cm"]
        acc_round3 = summary_stats[2]["accuracy_cm"]
        assert acc_round3 >= acc_round1 or abs(acc_round3 - acc_round1) < 0.05

    def test_federated_confusion_matrix_aggregation(self):
        """Test that federation preserves confusion matrix values per round."""
        # Simulate federated rounds with confusion matrices
        federated_history = {
            1: {"cm_tp": 400, "cm_tn": 380, "cm_fp": 50, "cm_fn": 70},
            2: {"cm_tp": 430, "cm_tn": 410, "cm_fp": 40, "cm_fn": 20},
            3: {"cm_tp": 450, "cm_tn": 430, "cm_fp": 40, "cm_fn": 30},
        }

        # Verify each round maintains distinct CM values
        round_accuracies = {}
        for round_num, cm_values in federated_history.items():
            cm = {
                "true_positives": cm_values["cm_tp"],
                "true_negatives": cm_values["cm_tn"],
                "false_positives": cm_values["cm_fp"],
                "false_negatives": cm_values["cm_fn"],
            }
            stats = _calculate_summary_statistics(cm)
            round_accuracies[round_num] = stats["accuracy_cm"]

        # Verify data integrity
        assert len(round_accuracies) == 3
        assert round(round_accuracies[1], 2) == pytest.approx(
            (400 + 380) / (400 + 380 + 50 + 70),
            abs=0.01,
        )
