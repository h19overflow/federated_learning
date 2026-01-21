"""
Unit tests for centralized training task functions.

Tests cover:
- Background task execution
- Trainer initialization and execution
- Error handling
- Result processing
"""

from unittest.mock import MagicMock, patch

import pytest

from federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks import (
    run_centralized_training_task,
)


class TestRunCentralizedTrainingTask:
    """Test run_centralized_training_task function."""

    @pytest.mark.unit
    def test_centralized_training_task_success(
        self,
        mock_centralized_trainer,
        tmp_path,
    ):
        """Test successful centralized training task."""
        source_path = str(tmp_path / "data")
        os.makedirs(source_path)

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
            return_value=mock_centralized_trainer,
        ):
            result = run_centralized_training_task(
                source_path=source_path,
                checkpoint_dir=str(tmp_path / "checkpoints"),
                logs_dir=str(tmp_path / "logs"),
                experiment_name="test_experiment",
                csv_filename="test.csv",
            )

            # Verify trainer was created and train was called
            assert "final_metrics" in result
            assert result["final_metrics"]["accuracy"] == 0.85

    @pytest.mark.unit
    def test_centralized_training_task_with_mock_config(self, tmp_path):
        """Test centralized training with mocked config path."""
        source_path = str(tmp_path / "data")
        os.makedirs(source_path)

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "final_metrics": {
                "train_loss": 0.3,
                "val_loss": 0.25,
                "accuracy": 0.92,
            },
        }

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
            return_value=mock_trainer,
        ):
            result = run_centralized_training_task(
                source_path=source_path,
                checkpoint_dir=str(tmp_path / "checkpoints"),
                logs_dir=str(tmp_path / "logs"),
                experiment_name="test_exp",
                csv_filename="data.csv",
            )

            assert result["final_metrics"]["accuracy"] == 0.92
            mock_trainer.train.assert_called_once()

    @pytest.mark.unit
    def test_centralized_training_task_trainer_error(self, tmp_path):
        """Test error handling when trainer raises exception."""
        source_path = str(tmp_path / "data")
        os.makedirs(source_path)

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = RuntimeError("Training failed")

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
            return_value=mock_trainer,
        ):
            result = run_centralized_training_task(
                source_path=source_path,
                checkpoint_dir=str(tmp_path / "checkpoints"),
                logs_dir=str(tmp_path / "logs"),
                experiment_name="test_exp",
                csv_filename="data.csv",
            )

            # Verify error is captured in result
            assert result["status"] == "failed"
            assert "error" in result
            assert "RuntimeError" in result["error"]

    @pytest.mark.unit
    def test_centralized_training_task_with_checkpoint_results(
        self,
        mock_centralized_trainer,
        tmp_path,
    ):
        """Test task with checkpoint path in results."""
        source_path = str(tmp_path / "data")
        os.makedirs(source_path)

        mock_trainer.train.return_value = {
            "final_metrics": {"loss": 0.4},
            "checkpoint_path": str(tmp_path / "checkpoints" / "model.ckpt"),
            "best_model_path": str(tmp_path / "checkpoints" / "best.ckpt"),
        }

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
            return_value=mock_trainer,
        ):
            result = run_centralized_training_task(
                source_path=source_path,
                checkpoint_dir=str(tmp_path / "checkpoints"),
                logs_dir=str(tmp_path / "logs"),
                experiment_name="test_exp",
                csv_filename="data.csv",
            )

            assert "checkpoint_path" in result
            assert "best_model_path" in result

    @pytest.mark.unit
    def test_centralized_training_task_various_experiments(self, tmp_path):
        """Test task with different experiment names."""
        source_path = str(tmp_path / "data")
        os.makedirs(source_path)

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "final_metrics": {"accuracy": 0.88},
        }

        experiment_names = [
            "exp_001",
            "baseline_test",
            "experiment_2025_01_21",
        ]

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
            return_value=mock_trainer,
        ):
            for exp_name in experiment_names:
                result = run_centralized_training_task(
                    source_path=source_path,
                    checkpoint_dir=str(tmp_path / "checkpoints"),
                    logs_dir=str(tmp_path / "logs"),
                    experiment_name=exp_name,
                    csv_filename="data.csv",
                )
                assert result["final_metrics"]["accuracy"] == 0.88

    @pytest.mark.unit
    def test_centralized_training_task_file_not_found_error(self, tmp_path):
        """Test handling of FileNotFoundError from trainer."""
        source_path = str(tmp_path / "data")
        os.makedirs(source_path)

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = FileNotFoundError("CSV not found")

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
            return_value=mock_trainer,
        ):
            result = run_centralized_training_task(
                source_path=source_path,
                checkpoint_dir=str(tmp_path / "checkpoints"),
                logs_dir=str(tmp_path / "logs"),
                experiment_name="test_exp",
                csv_filename="data.csv",
            )

            assert result["status"] == "failed"
            assert "FileNotFoundError" in result["error"]

    @pytest.mark.unit
    def test_centralized_training_task_value_error(self, tmp_path):
        """Test handling of ValueError from trainer."""
        source_path = str(tmp_path / "data")
        os.makedirs(source_path)

        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = ValueError("Invalid configuration")

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
            return_value=mock_trainer,
        ):
            result = run_centralized_training_task(
                source_path=source_path,
                checkpoint_dir=str(tmp_path / "checkpoints"),
                logs_dir=str(tmp_path / "logs"),
                experiment_name="test_exp",
                csv_filename="data.csv",
            )

            assert result["status"] == "failed"
            assert "ValueError" in result["error"]

    @pytest.mark.unit
    def test_centralized_training_task_with_empty_metrics(self, tmp_path):
        """Test task with empty metrics dictionary."""
        source_path = str(tmp_path / "data")
        os.makedirs(source_path)

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "final_metrics": {},
        }

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
            return_value=mock_trainer,
        ):
            result = run_centralized_training_task(
                source_path=source_path,
                checkpoint_dir=str(tmp_path / "checkpoints"),
                logs_dir=str(tmp_path / "logs"),
                experiment_name="test_exp",
                csv_filename="data.csv",
            )

            assert "final_metrics" in result
            assert result["final_metrics"] == {}

    @pytest.mark.unit
    def test_centralized_training_task_logs_training_start(
        self,
        mock_centralized_trainer,
        tmp_path,
        caplog,
    ):
        """Test that task logs training start appropriately."""
        import logging

        caplog.set_level(logging.INFO)

        source_path = str(tmp_path / "data")
        os.makedirs(source_path)

        mock_trainer.train.return_value = {
            "final_metrics": {"accuracy": 0.90},
        }

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
            return_value=mock_trainer,
        ):
            run_centralized_training_task(
                source_path=source_path,
                checkpoint_dir=str(tmp_path / "checkpoints"),
                logs_dir=str(tmp_path / "logs"),
                experiment_name="test_exp",
                csv_filename="data.csv",
            )

            # Check that training was logged
            log_messages = [record.message for record in caplog.records]
            assert any("CENTRALIZED TRAINING" in msg for msg in log_messages)
            assert any("TRAINING COMPLETED" in msg for msg in log_messages)
