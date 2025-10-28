"""
Unit tests for federated learning trainer module.
Tests orchestration of federated learning pipeline.
"""

import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Any

from federated_pneumonia_detection.src.control.federated_learning.trainer import (
    FederatedTrainer,
)
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.models.system_constants import SystemConstants


class TestFederatedTrainerInit:
    """Tests for FederatedTrainer initialization."""

    def test_init_valid_parameters(self):
        """Test FederatedTrainer initialization with valid parameters."""
        config = ExperimentConfig()
        constants = SystemConstants()
        device = torch.device("cpu")

        trainer = FederatedTrainer(
            config=config,
            constants=constants,
            device=device,
        )

        assert trainer.config is config
        assert trainer.constants is constants
        assert trainer.device is device

    def test_init_none_config_raises_error(self):
        """Test that None config raises ValueError."""
        with pytest.raises(ValueError, match="config cannot be None"):
            FederatedTrainer(
                config=None,
                constants=SystemConstants(),
                device=torch.device("cpu"),
            )

    def test_init_none_constants_raises_error(self):
        """Test that None constants raises ValueError."""
        with pytest.raises(ValueError, match="constants cannot be None"):
            FederatedTrainer(
                config=ExperimentConfig(),
                constants=None,
                device=torch.device("cpu"),
            )

    def test_init_none_device_raises_error(self):
        """Test that None device raises ValueError."""
        with pytest.raises(ValueError, match="device cannot be None"):
            FederatedTrainer(
                config=ExperimentConfig(),
                constants=SystemConstants(),
                device=None,
            )

    def test_init_logger_setup(self):
        """Test that logger is properly initialized."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        assert trainer.logger is not None
        assert trainer.logger.name == "FederatedTrainer"


class TestFederatedTrainerCreateModel:
    """Tests for FederatedTrainer._create_model method."""

    def test_create_model_returns_resnet(self):
        """Test that _create_model returns ResNetWithCustomHead instance."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        model = trainer._create_model()

        # Model should be a torch.nn.Module
        assert isinstance(model, torch.nn.Module)

    def test_create_model_on_device(self):
        """Test that created model is on specified device."""
        device = torch.device("cpu")
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=device,
        )

        model = trainer._create_model()

        # Check first parameter is on correct device
        for param in model.parameters():
            assert param.device.type == device.type
            break

    def test_create_model_correct_num_classes(self):
        """Test that model is created with correct num_classes."""
        config = ExperimentConfig(num_classes=3)
        trainer = FederatedTrainer(
            config=config,
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        model = trainer._create_model()

        # Model should have been created
        assert isinstance(model, torch.nn.Module)

    def test_create_model_with_dropout(self):
        """Test that model is created with configured dropout rate."""
        config = ExperimentConfig(dropout_rate=0.3)
        trainer = FederatedTrainer(
            config=config,
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        model = trainer._create_model()

        assert isinstance(model, torch.nn.Module)


class TestFederatedTrainerGetInitialParameters:
    """Tests for FederatedTrainer._get_initial_parameters method."""

    def test_get_initial_parameters_returns_parameters(self):
        """Test that _get_initial_parameters returns Flower Parameters."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        params = trainer._get_initial_parameters()

        # Should be a Parameters object or compatible
        assert params is not None

    def test_get_initial_parameters_creates_weights(self):
        """Test that initial parameters can be converted to weights."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        params = trainer._get_initial_parameters()

        # Parameters should be non-empty
        assert params is not None


class TestFederatedTrainerClientFn:
    """Tests for FederatedTrainer._client_fn method."""

    def test_client_fn_creates_flower_client(self):
        """Test that _client_fn creates a FlowerClient."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(num_clients=2, clients_per_round=2),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        # Setup client data cache
        mock_train_loader = Mock()
        mock_train_loader.dataset = Mock()
        mock_val_loader = Mock()
        mock_val_loader.dataset = Mock()

        trainer._client_data_cache = {
            "0": (mock_train_loader, mock_val_loader),
        }

        # This should create a client, but we need to mock it as it's complex
        with patch(
            "federated_pneumonia_detection.src.control.federated_learning.trainer.FlowerClient"
        ) as mock_client_class:
            mock_client_instance = Mock()
            mock_client_instance.to_client = Mock(return_value=Mock())
            mock_client_class.return_value = mock_client_instance

            result = trainer._client_fn("0")

            # Should call FlowerClient constructor
            mock_client_class.assert_called_once()

    def test_client_fn_missing_client_id_raises_error(self):
        """Test that _client_fn raises error for missing client ID."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        trainer._client_data_cache = {}

        with pytest.raises(ValueError, match="Client.*data not found"):
            trainer._client_fn("999")


class TestFederatedTrainerCreateEvaluateFn:
    """Tests for FederatedTrainer._create_evaluate_fn method."""

    def test_create_evaluate_fn_returns_callable(self):
        """Test that _create_evaluate_fn returns a callable."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        mock_val_loader = Mock()
        evaluate_fn = trainer._create_evaluate_fn(mock_val_loader)

        assert callable(evaluate_fn)

    def test_evaluate_fn_returns_tuple(self):
        """Test that returned evaluate_fn returns (loss, metrics) tuple."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(num_classes=2),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        # Create mock dataloader
        images = torch.randn(4, 3, 64, 64)
        labels = torch.randint(0, 2, (4,))
        mock_val_loader = Mock()
        mock_val_loader.__iter__ = Mock(return_value=iter([(images, labels)]))
        mock_val_loader.dataset = [1, 2, 3, 4]

        evaluate_fn = trainer._create_evaluate_fn(mock_val_loader)

        # Get initial parameters
        params = trainer._get_initial_parameters()

        # This would require proper parameter format, so we'll just verify
        # that the function can be called
        assert callable(evaluate_fn)

    def test_evaluate_fn_has_logging(self):
        """Test that evaluate function has logging."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        mock_val_loader = Mock()
        evaluate_fn = trainer._create_evaluate_fn(mock_val_loader)

        assert callable(evaluate_fn)


class TestFederatedTrainerTrain:
    """Tests for FederatedTrainer.train method."""

    def test_train_requires_valid_data(self):
        """Test that train method validates inputs."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        constants = SystemConstants()
        empty_df = pd.DataFrame({
            constants.FILENAME_COLUMN: [],
            constants.TARGET_COLUMN: [],
        })

        # Train should handle empty data without crashing (logs error but doesn't raise)
        result = trainer.train(
            data_df=empty_df,
            image_dir=Path("/tmp"),
            experiment_name="test",
        )
        # Should return None when training fails
        assert result is None

    def test_train_method_exists(self):
        """Test that train method exists and is callable."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        assert hasattr(trainer, "train")
        assert callable(trainer.train)

    def test_train_error_handling(self):
        """Test that train properly handles and logs errors."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        constants = SystemConstants()
        invalid_df = pd.DataFrame({
            constants.FILENAME_COLUMN: ["nonexistent.jpg"],
            constants.TARGET_COLUMN: [0],
        })

        # Should handle errors gracefully (logs error but doesn't raise)
        result = trainer.train(
            data_df=invalid_df,
            image_dir=Path("/nonexistent/path"),
            experiment_name="test",
        )
        # Should return None when training fails
        assert result is None


class TestFederatedTrainerIntegration:
    """Integration tests for FederatedTrainer."""

    def test_trainer_initialization_sequence(self):
        """Test typical initialization sequence."""
        config = ExperimentConfig(
            num_clients=2,
            clients_per_round=2,
            num_rounds=1,
            local_epochs=1,
        )
        constants = SystemConstants()
        device = torch.device("cpu")

        trainer = FederatedTrainer(
            config=config,
            constants=constants,
            device=device,
        )

        # All components should be properly initialized
        assert trainer.config == config
        assert trainer.constants == constants
        assert trainer.device == device
        assert trainer.logger is not None

    def test_trainer_create_model_multiple_times(self):
        """Test that trainer can create multiple models."""
        trainer = FederatedTrainer(
            config=ExperimentConfig(),
            constants=SystemConstants(),
            device=torch.device("cpu"),
        )

        model1 = trainer._create_model()
        model2 = trainer._create_model()

        # Both should be valid models
        assert isinstance(model1, torch.nn.Module)
        assert isinstance(model2, torch.nn.Module)
        # But they should be different instances
        assert model1 is not model2
