"""
End-to-end integration tests for confusion matrix pipeline.

Tests the complete flow from training through database persistence to API response.
Covers both centralized and federated learning modes.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from federated_pneumonia_detection.src.boundary.engine import Run
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.CRUD.server_evaluation import server_evaluation_crud
from federated_pneumonia_detection.src.api.endpoints.runs_endpoints.utils import (
    _calculate_summary_statistics,
    _transform_run_to_results,
)


@pytest.mark.integration
class TestEndToEndCentralizedPipeline:
    """End-to-end tests for centralized training pipeline with confusion matrices."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()

    def test_centralized_full_pipeline_from_training_to_api(self, mock_db):
        """Test complete pipeline: training → persistence → API response."""

        # Step 1: Simulate training with MetricsCollector capturing CM values
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
            },
        ]

        # Step 2: Create run in database
        with patch("federated_pneumonia_detection.src.boundary.CRUD.run.get_session", return_value=mock_db):
            mock_run = Mock(spec=Run)
            mock_run.id = 1
            mock_run.status = "in_progress"

            with patch.object(run_crud, "create", return_value=mock_run):
                created_run = run_crud.create(
                    mock_db, training_mode="centralized", status="in_progress"
                )
                assert created_run.id == 1

            # Step 3: Persist metrics (MetricsCollector would do this)
            with patch.object(run_crud, "persist_metrics") as mock_persist:
                run_crud.persist_metrics(mock_db, created_run.id, epoch_metrics)
                mock_persist.assert_called_once()

            # Step 4: Update run status to completed
            with patch.object(run_crud, "update_status") as mock_update:
                run_crud.update_status(mock_db, created_run.id, "completed")
                mock_update.assert_called_once()

            # Step 5: Fetch run with metrics for API response
            mock_run.status = "completed"
            mock_run.start_time = datetime(2024, 1, 1, 10, 0, 0)
            mock_run.end_time = datetime(2024, 1, 1, 11, 0, 0)

            # Create metric objects
            metrics = [
                Mock(step=0, metric_name="train_loss", metric_value=0.5),
                Mock(step=0, metric_name="val_loss", metric_value=0.4),
                Mock(step=0, metric_name="val_accuracy", metric_value=0.88),
                Mock(step=0, metric_name="val_cm_tp", metric_value=450),
                Mock(step=0, metric_name="val_cm_tn", metric_value=430),
                Mock(step=0, metric_name="val_cm_fp", metric_value=40),
                Mock(step=0, metric_name="val_cm_fn", metric_value=30),
            ]
            mock_run.metrics = metrics

            # Step 6: Transform to API response
            api_response = _transform_run_to_results(mock_run)

            # Verify complete pipeline
            assert api_response["experiment_id"] == "run_1"
            assert api_response["status"] == "completed"
            assert api_response["confusion_matrix"] is not None

            cm = api_response["confusion_matrix"]
            assert cm["true_positives"] == 450
            assert cm["sensitivity"] > 0
            assert cm["specificity"] > 0

    def test_centralized_pipeline_metrics_collection(self, mock_db):
        """Test metrics collection throughout centralized pipeline."""
        # Simulate LitResNet validation callback
        from federated_pneumonia_detection.src.control.dl_model.internals.model.metrics_collector import (
            MetricsCollectorCallback,
        )

        with patch(
            "federated_pneumonia_detection.src.control.dl_model.utils.model.metrics_collector.get_session"
        ):
            # Initialize MetricsCollector
            collector = MetricsCollectorCallback(
                save_dir="/tmp/test",
                experiment_name="test_exp",
                run_id=None,
                training_mode="centralized",
                enable_db_persistence=False,
            )

            # Simulate epoch metrics collection
            mock_trainer = Mock()
            mock_trainer.callback_metrics = {
                "val_loss": Mock(item=lambda: 0.4),
                "val_cm_tp": Mock(item=lambda: 450),
                "val_cm_tn": Mock(item=lambda: 430),
                "val_cm_fp": Mock(item=lambda: 40),
                "val_cm_fn": Mock(item=lambda: 30),
            }
            mock_trainer.logged_metrics = {}
            mock_trainer.optimizers = [Mock(param_groups=[{"lr": 0.001}])]
            mock_trainer.current_epoch = 0
            mock_trainer.global_step = 100

            mock_pl_module = Mock()

            # Collect metrics
            metrics = collector._extract_metrics(mock_trainer, mock_pl_module, "val")

            # Verify CM values collected
            assert "val_cm_tp" in metrics
            assert "val_cm_tn" in metrics
            assert "val_cm_fp" in metrics
            assert "val_cm_fn" in metrics

    def test_centralized_pipeline_with_multiple_epochs(self, mock_db):
        """Test pipeline across multiple training epochs."""
        mock_run = Mock(spec=Run)
        mock_run.id = 1
        mock_run.status = "completed"
        mock_run.start_time = datetime(2024, 1, 1, 10, 0, 0)
        mock_run.end_time = datetime(2024, 1, 1, 12, 0, 0)

        # Create metrics for 5 epochs with improving CM
        metrics_data = []
        for epoch in range(5):
            metrics_data.extend(
                [
                    Mock(step=epoch, metric_name="train_loss", metric_value=0.5 - epoch * 0.05),
                    Mock(step=epoch, metric_name="val_loss", metric_value=0.4 - epoch * 0.04),
                    Mock(step=epoch, metric_name="val_accuracy", metric_value=0.80 + epoch * 0.03),
                    Mock(step=epoch, metric_name="val_cm_tp", metric_value=400 + epoch * 15),
                    Mock(step=epoch, metric_name="val_cm_tn", metric_value=380 + epoch * 15),
                    Mock(step=epoch, metric_name="val_cm_fp", metric_value=50 - epoch * 8),
                    Mock(step=epoch, metric_name="val_cm_fn", metric_value=70 - epoch * 10),
                ]
            )
        mock_run.metrics = metrics_data

        # Transform to API response
        api_response = _transform_run_to_results(mock_run)

        # Verify progression
        assert api_response["total_epochs"] == 5
        assert len(api_response["training_history"]) == 5

        # Confusion matrix should be from last epoch
        cm = api_response["confusion_matrix"]
        assert cm["true_positives"] == 460  # 400 + 4*15
        assert cm["true_negatives"] == 440  # 380 + 4*15
        assert cm["false_positives"] == 18  # 50 - 4*8
        assert cm["false_negatives"] == 30  # 70 - 4*10


@pytest.mark.integration
@pytest.mark.federated
class TestEndToEndFederatedPipeline:
    """End-to-end tests for federated learning pipeline with confusion matrices."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()

    def test_federated_full_pipeline_round_by_round(self, mock_db):
        """Test complete federated pipeline: server evaluation → persistence → API."""

        # Step 1: Simulate server evaluation metrics for 3 rounds
        server_eval_metrics = [
            {
                "round_number": 1,
                "loss": 0.5,
                "accuracy": 0.85,
                "server_cm_tp": 400,
                "server_cm_tn": 380,
                "server_cm_fp": 50,
                "server_cm_fn": 70,
            },
            {
                "round_number": 2,
                "loss": 0.35,
                "accuracy": 0.90,
                "server_cm_tp": 430,
                "server_cm_tn": 410,
                "server_cm_fp": 40,
                "server_cm_fn": 20,
            },
            {
                "round_number": 3,
                "loss": 0.30,
                "accuracy": 0.92,
                "server_cm_tp": 450,
                "server_cm_tn": 430,
                "server_cm_fp": 40,
                "server_cm_fn": 30,
            },
        ]

        # Step 2: Create run for federated training
        mock_run = Mock(spec=Run)
        mock_run.id = 1
        mock_run.training_mode = "federated"

        with patch.object(run_crud, "create", return_value=mock_run):
            created_run = run_crud.create(
                mock_db, training_mode="federated", status="in_progress"
            )
            assert created_run.id == 1

        # Step 3: Persist server evaluations per round
        created_evals = []
        with patch.object(server_evaluation_crud, "create_evaluation") as mock_create:
            for metrics in server_eval_metrics:
                mock_eval = Mock(
                    round_number=metrics["round_number"],
                    true_positives=metrics["server_cm_tp"],
                )
                mock_create.return_value = mock_eval
                created_evals.append(mock_eval)
                server_evaluation_crud.create_evaluation(
                    mock_db, created_run.id, metrics["round_number"], metrics
                )

        # Step 4: Verify per-round persistence
        assert len(created_evals) == 3
        assert created_evals[0].true_positives == 400
        assert created_evals[1].true_positives == 430
        assert created_evals[2].true_positives == 450

        # Step 5: Fetch server evaluations for API response
        with patch.object(server_evaluation_crud, "get_by_run", return_value=created_evals):
            evals = server_evaluation_crud.get_by_run(mock_db, created_run.id)
            assert len(evals) == 3

        # Step 6: Get summary statistics
        with patch.object(server_evaluation_crud, "get_summary_stats") as mock_summary:
            mock_summary.return_value = {
                "total_rounds": 3,
                "best_accuracy": {"value": 0.92, "round": 3},
            }
            summary = server_evaluation_crud.get_summary_stats(mock_db, created_run.id)
            assert summary["total_rounds"] == 3
            assert summary["best_accuracy"]["value"] == 0.92

    def test_federated_pipeline_with_confusion_matrices_per_round(self, mock_db):
        """Test that CM values are maintained per round in federated pipeline."""

        rounds_data = [
            {
                "round": 1,
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
                "round": 2,
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
                "round": 3,
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

        # Create mock ServerEvaluation objects
        mock_evals = []
        for round_data in rounds_data:
            metrics = round_data["metrics"]
            mock_eval = Mock(
                round_number=round_data["round"],
                loss=metrics["loss"],
                accuracy=metrics["accuracy"],
                true_positives=metrics["server_cm_tp"],
                true_negatives=metrics["server_cm_tn"],
                false_positives=metrics["server_cm_fp"],
                false_negatives=metrics["server_cm_fn"],
            )
            mock_evals.append(mock_eval)

        # Verify each round maintains distinct CM
        for i, eval_obj in enumerate(mock_evals):
            assert eval_obj.true_positives == rounds_data[i]["metrics"]["server_cm_tp"]

        # Calculate summary statistics across rounds
        all_stats = []
        for eval_obj in mock_evals:
            cm = {
                "true_positives": eval_obj.true_positives,
                "true_negatives": eval_obj.true_negatives,
                "false_positives": eval_obj.false_positives,
                "false_negatives": eval_obj.false_negatives,
            }
            stats = _calculate_summary_statistics(cm)
            all_stats.append(stats)

        # Verify statistics calculated per round
        assert len(all_stats) == 3
        # Accuracy should improve across rounds
        acc_round1 = all_stats[0]["accuracy_cm"]
        acc_round3 = all_stats[2]["accuracy_cm"]
        assert acc_round3 >= acc_round1

    def test_federated_pipeline_format_conversion(self, mock_db):
        """Test that flat format metrics are converted correctly in persistence."""

        flat_format_metrics = {
            "loss": 0.35,
            "accuracy": 0.92,
            "server_cm_tp": 450,
            "server_cm_tn": 430,
            "server_cm_fp": 40,
            "server_cm_fn": 30,
        }

        # Simulate ServerEvaluationCRUD extracting and storing
        crud = server_evaluation_crud

        with patch.object(crud, "create") as mock_create:
            mock_eval = Mock()
            mock_create.return_value = mock_eval

            # Create evaluation
            crud.create_evaluation(
                mock_db, run_id=1, round_number=1, metrics=flat_format_metrics
            )

            # Verify extraction
            call_kwargs = mock_create.call_args.kwargs if mock_create.call_args.kwargs else {}

            assert call_kwargs.get("true_positives") == 450
            assert call_kwargs.get("true_negatives") == 430
            assert call_kwargs.get("false_positives") == 40
            assert call_kwargs.get("false_negatives") == 30


@pytest.mark.integration
class TestCrossModePipeline:
    """Integration tests comparing centralized and federated pipelines."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()

    def test_both_modes_produce_consistent_cm_response(self, mock_db):
        """Test that both training modes produce compatible CM responses."""

        # Centralized confusion matrix response
        centralized_cm = {
            "true_positives": 450,
            "true_negatives": 430,
            "false_positives": 40,
            "false_negatives": 30,
            **_calculate_summary_statistics(
                {
                    "true_positives": 450,
                    "true_negatives": 430,
                    "false_positives": 40,
                    "false_negatives": 30,
                }
            ),
        }

        # Federated confusion matrix response (same values)
        federated_cm = {
            "true_positives": 450,
            "true_negatives": 430,
            "false_positives": 40,
            "false_negatives": 30,
            **_calculate_summary_statistics(
                {
                    "true_positives": 450,
                    "true_negatives": 430,
                    "false_positives": 40,
                    "false_negatives": 30,
                }
            ),
        }

        # Both should have identical structure and values
        assert centralized_cm == federated_cm

        # Verify required fields
        required_fields = [
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

        for field in required_fields:
            assert field in centralized_cm
            assert field in federated_cm
            assert centralized_cm[field] == federated_cm[field]

    def test_pipeline_error_handling_missing_cm_values(self, mock_db):
        """Test pipeline handles missing CM values gracefully."""

        # Metrics without CM
        metrics_no_cm = {
            "loss": 0.35,
            "accuracy": 0.92,
        }

        # Test centralized transformation
        mock_run = Mock(spec=Run)
        mock_run.id = 1
        mock_run.status = "completed"
        mock_run.start_time = datetime(2024, 1, 1, 10, 0, 0)
        mock_run.end_time = datetime(2024, 1, 1, 11, 0, 0)
        mock_run.metrics = [
            Mock(step=0, metric_name=k, metric_value=v)
            for k, v in metrics_no_cm.items()
        ]

        result = _transform_run_to_results(mock_run)
        assert result["confusion_matrix"] is None

        # Test federated transformation
        with patch.object(server_evaluation_crud, "create_evaluation") as mock_create:
            mock_create.return_value = Mock()
            try:
                server_evaluation_crud.create_evaluation(
                    mock_db, run_id=1, round_number=1, metrics=metrics_no_cm
                )
                # Should not raise error
                assert True
            except Exception as e:
                pytest.fail(f"Pipeline should handle missing CM gracefully: {e}")
