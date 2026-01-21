"""
Unit tests for CentralizedTrainer module.
Tests orchestration of centralized training pipeline with mocked components.
"""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from federated_pneumonia_detection.src.control.dl_model.centralized_trainer import (
    CentralizedTrainer,
)


class TestCentralizedTrainer:
    """Tests for CentralizedTrainer."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.get.side_effect = lambda key, default=None: {
            "experiment.epochs": 10,
            "experiment.batch_size": 32,
            "experiment.learning_rate": 0.001,
            "experiment.weight_decay": 1e-4,
            "experiment.fine_tune_layers_count": 2,
            "experiment.validation_split": 0.2,
            "experiment.seed": 42,
            "columns.target": "Target",
            "system.seed": 42,
        }.get(key, default)
        return config

    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.ConfigManager",
    )
    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.DataSourceExtractor",
    )
    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.os.makedirs",
    )
    def test_init(self, mock_makedirs, mock_extractor, mock_config_class, mock_config):
        """Test initialization of CentralizedTrainer."""
        mock_config_class.return_value = mock_config

        trainer = CentralizedTrainer(config_path="fake_config.yaml")

        assert trainer.checkpoint_dir == "results/checkpoints"
        assert trainer.logs_dir == "results/training_logs"
        mock_makedirs.assert_any_call("results/checkpoints", exist_ok=True)
        mock_makedirs.assert_any_call("results/training_logs", exist_ok=True)
        mock_extractor.assert_called_once()

    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.ConfigManager",
    )
    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.get_session",
    )
    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.run_crud",
    )
    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.load_metadata",
    )
    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.create_train_val_split",
    )
    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.XRayDataModule",
    )
    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.LitResNetEnhanced",
    )
    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.create_trainer_from_config",
    )
    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.TensorBoardLogger",
    )
    def test_train_workflow(
        self,
        mock_tb_logger,
        mock_create_trainer,
        mock_model_class,
        mock_datamodule_class,
        mock_split,
        mock_load,
        mock_run_crud,
        mock_get_session,
        mock_config_class,
        mock_config,
    ):
        """Test the complete training workflow of CentralizedTrainer."""
        # Setup mocks
        mock_config_class.return_value = mock_config
        mock_db = MagicMock()
        mock_get_session.return_value = mock_db
        mock_run_crud.create.return_value = Mock(id=1)

        mock_load.return_value = pd.DataFrame({"Target": [0, 1]})
        mock_split.return_value = (
            pd.DataFrame({"Target": [0]}),
            pd.DataFrame({"Target": [1]}),
        )

        mock_trainer = MagicMock()
        mock_create_trainer.return_value = mock_trainer
        mock_trainer.callbacks = []
        mock_trainer.current_epoch = 10
        mock_trainer.global_step = 100
        mock_trainer.state.stage.value = "finished"

        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        mock_model.get_model_summary.return_value = "summary"

        # Initialize trainer
        trainer = CentralizedTrainer()
        trainer.data_source_extractor = MagicMock()
        trainer.data_source_extractor.extract_and_validate.return_value = (
            "img_dir",
            "csv_path",
        )

        # Call train
        results = trainer.train(source_path="fake.zip", experiment_name="test_exp")

        # Assertions
        assert results["run_id"] == 1
        assert results["current_epoch"] == 10
        mock_create_trainer.assert_called_once()
        # Verify trainer fit was called with model and datamodule
        mock_trainer.fit.assert_called_once_with(
            mock_model,
            mock_datamodule_class.return_value,
        )

        # Verify database interaction
        mock_run_crud.create.assert_called_once()
        mock_run_crud.complete_run.assert_called_once_with(
            mock_db,
            run_id=1,
            status="completed",
        )

    @patch(
        "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.ConfigManager",
    )
    def test_create_data_module(self, mock_config_class, mock_config):
        """Test _create_data_module logic."""
        mock_config_class.return_value = mock_config
        trainer = CentralizedTrainer()

        train_df = pd.DataFrame({"Target": [0]})
        val_df = pd.DataFrame({"Target": [1]})
        image_dir = "fake_dir"

        with patch(
            "federated_pneumonia_detection.src.control.dl_model.centralized_trainer.XRayDataModule",
        ) as mock_dm:
            trainer._create_data_module(train_df, val_df, image_dir)
            mock_dm.assert_called_once_with(
                train_df=train_df,
                val_df=val_df,
                config=mock_config,
                image_dir=image_dir,
                validate_images_on_init=False,
            )
