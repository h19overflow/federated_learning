"""
Unit tests for Flower client and server apps.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch

from federated_pneumonia_detection.src.control.federated_learning._old_reference import server_app, client_app


@pytest.mark.unit
class TestServerApp:
    """Test server application functions."""

    @patch('federated_pneumonia_detection.src.control.federated_learning.server_app.ConfigLoader')
    def test_load_server_config(self, mock_config_loader):
        """Test server configuration loading."""
        mock_loader_instance = MagicMock()
        mock_constants = Mock()
        mock_config = Mock()
        
        mock_loader_instance.create_system_constants.return_value = mock_constants
        mock_loader_instance.create_experiment_config.return_value = mock_config
        mock_config_loader.return_value = mock_loader_instance

        constants, config = server_app.load_server_config()

        assert constants is not None
        assert config is not None
        mock_loader_instance.create_system_constants.assert_called_once()
        mock_loader_instance.create_experiment_config.assert_called_once()

    @patch('federated_pneumonia_detection.src.control.federated_learning.server_app.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.server_app.ResNetWithCustomHead')
    def test_create_global_model(self, mock_model_class, mock_config_loader):
        """Test global model creation."""
        # Setup mocks
        mock_loader_instance = MagicMock()
        mock_constants = Mock()
        mock_config = Mock()
        mock_config.num_classes = 2
        mock_config.dropout_rate = 0.3
        mock_config.fine_tune_layers_count = 2
        
        mock_loader_instance.create_system_constants.return_value = mock_constants
        mock_loader_instance.create_experiment_config.return_value = mock_config
        mock_config_loader.return_value = mock_loader_instance

        mock_model = MagicMock()
        mock_model.get_model_info.return_value = {
            'total_parameters': 1000000,
            'trainable_parameters': 500000
        }
        mock_model_class.return_value = mock_model

        model = server_app.create_global_model(mock_constants, mock_config)

        assert model is not None
        mock_model_class.assert_called_once()


@pytest.mark.unit
class TestClientApp:
    """Test client application functions."""

    @patch('federated_pneumonia_detection.src.control.federated_learning.client_app.ConfigLoader')
    def test_load_client_config(self, mock_config_loader):
        """Test client configuration loading."""
        mock_loader_instance = MagicMock()
        mock_constants = Mock()
        mock_config = Mock()
        
        mock_loader_instance.create_system_constants.return_value = mock_constants
        mock_loader_instance.create_experiment_config.return_value = mock_config
        mock_config_loader.return_value = mock_loader_instance

        constants, config = client_app.load_client_config()

        assert constants is not None
        assert config is not None

    @patch('federated_pneumonia_detection.src.control.federated_learning.client_app.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.client_app.ResNetWithCustomHead')
    def test_create_client_model(self, mock_model_class, mock_config_loader):
        """Test client model creation."""
        mock_loader_instance = MagicMock()
        mock_constants = Mock()
        mock_config = Mock()
        mock_config.num_classes = 2
        mock_config.dropout_rate = 0.3
        mock_config.fine_tune_layers_count = 2
        
        mock_loader_instance.create_system_constants.return_value = mock_constants
        mock_loader_instance.create_experiment_config.return_value = mock_config
        mock_config_loader.return_value = mock_loader_instance

        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        model = client_app.create_client_model(mock_constants, mock_config)

        assert model is not None
        mock_model_class.assert_called_once_with(
            constants=mock_constants,
            config=mock_config,
            num_classes=mock_config.num_classes,
            dropout_rate=mock_config.dropout_rate,
            fine_tune_layers_count=mock_config.fine_tune_layers_count
        )


@pytest.mark.unit
@pytest.mark.federated
class TestFlowerIntegration:
    """Test Flower app integration points."""

    def test_server_app_exists(self):
        """Test that server app is properly initialized."""
        from federated_pneumonia_detection.src.control.federated_learning.server_app import app
        assert app is not None

    def test_client_app_exists(self):
        """Test that client app is properly initialized."""
        from federated_pneumonia_detection.src.control.federated_learning.client_app import app
        assert app is not None

    @patch('federated_pneumonia_detection.src.control.federated_learning.server_app.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.server_app.ResNetWithCustomHead')
    def test_server_config_params_used(self, mock_model, mock_config_loader):
        """Test that server uses config parameters correctly."""
        mock_loader_instance = MagicMock()
        mock_constants = Mock()
        mock_config = Mock()
        mock_config.num_clients = 5
        mock_config.clients_per_round = 3
        mock_config.num_rounds = 10
        
        mock_loader_instance.create_system_constants.return_value = mock_constants
        mock_loader_instance.create_experiment_config.return_value = mock_config
        mock_config_loader.return_value = mock_loader_instance

        constants, config = server_app.load_server_config()

        assert config.num_clients == 5
        assert config.clients_per_round == 3
        assert config.num_rounds == 10


