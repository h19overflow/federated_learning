"""
Unit tests for FederatedTrainer.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import tempfile
from pathlib import Path

from federated_pneumonia_detection.src.control.federated_learning.federated_trainer import FederatedTrainer
from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig


@pytest.mark.unit
class TestFederatedTrainerInit:
    """Test FederatedTrainer initialization."""

    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ZipHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DirectoryHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DatasetPreparer')
    def test_initialization_without_config(self, mock_preparer, mock_dir, mock_zip, mock_config_loader):
        """Test initialization without config path."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.create_system_constants.return_value = SystemConstants()
        mock_loader_instance.create_experiment_config.return_value = ExperimentConfig()
        mock_config_loader.return_value = mock_loader_instance

        trainer = FederatedTrainer()

        assert trainer.partition_strategy == "iid"
        assert trainer.checkpoint_dir == "federated_checkpoints"
        assert trainer.logs_dir == "federated_logs"

    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ZipHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DirectoryHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DatasetPreparer')
    def test_initialization_with_custom_dirs(self, mock_preparer, mock_dir, mock_zip, mock_config_loader):
        """Test initialization with custom directories."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.create_system_constants.return_value = SystemConstants()
        mock_loader_instance.create_experiment_config.return_value = ExperimentConfig()
        mock_config_loader.return_value = mock_loader_instance

        trainer = FederatedTrainer(
            checkpoint_dir="custom_checkpoints",
            logs_dir="custom_logs",
            partition_strategy="stratified"
        )

        assert trainer.partition_strategy == "stratified"
        assert trainer.checkpoint_dir == "custom_checkpoints"
        assert trainer.logs_dir == "custom_logs"


@pytest.mark.unit
class TestDataPartitioning:
    """Test data partitioning functionality."""

    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ZipHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DirectoryHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DatasetPreparer')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.partition_data_iid')
    def test_partition_iid(self, mock_partition_iid, mock_preparer, mock_dir, mock_zip, mock_config_loader):
        """Test IID partitioning is called correctly."""
        # Setup mocks with real config objects
        mock_loader_instance = MagicMock()
        mock_loader_instance.create_system_constants.return_value = SystemConstants()
        mock_loader_instance.create_experiment_config.return_value = ExperimentConfig(num_clients=3)
        mock_config_loader.return_value = mock_loader_instance
        
        trainer = FederatedTrainer(partition_strategy="iid")
        
        # Create test dataframe
        test_df = pd.DataFrame({
            'patientId': [f'p{i}' for i in range(30)],
            'Target': [0, 1] * 15
        })
        
        # Mock partition function
        mock_partition_iid.return_value = [
            test_df.iloc[:10],
            test_df.iloc[10:20],
            test_df.iloc[20:]
        ]
        
        result = trainer._partition_data_for_clients(test_df)
        
        assert len(result) == 3
        mock_partition_iid.assert_called_once()

    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ZipHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DirectoryHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DatasetPreparer')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.partition_data_stratified')
    def test_partition_stratified(self, mock_partition_strat, mock_preparer, mock_dir, mock_zip, mock_config_loader):
        """Test stratified partitioning is called correctly."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.create_system_constants.return_value = SystemConstants()
        mock_loader_instance.create_experiment_config.return_value = ExperimentConfig(num_clients=1, clients_per_round=1)
        mock_config_loader.return_value = mock_loader_instance
        
        trainer = FederatedTrainer(partition_strategy="stratified")
        
        test_df = pd.DataFrame({
            'patientId': [f'p{i}' for i in range(30)],
            'Target': [0, 1] * 15
        })
        
        mock_partition_strat.return_value = [test_df.iloc[:10]]
        
        result = trainer._partition_data_for_clients(test_df)
        
        mock_partition_strat.assert_called_once()

    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ZipHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DirectoryHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DatasetPreparer')
    def test_partition_invalid_strategy(self, mock_preparer, mock_dir, mock_zip, mock_config_loader):
        """Test error on invalid partition strategy."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.create_system_constants.return_value = SystemConstants()
        mock_loader_instance.create_experiment_config.return_value = ExperimentConfig()
        mock_config_loader.return_value = mock_loader_instance

        trainer = FederatedTrainer(partition_strategy="invalid")
        
        test_df = pd.DataFrame({
            'patientId': [f'p{i}' for i in range(30)],
            'Target': [0, 1] * 15
        })
        
        with pytest.raises(ValueError, match="Unknown partition strategy"):
            trainer._partition_data_for_clients(test_df)


@pytest.mark.unit
class TestTrainerMethods:
    """Test trainer utility methods."""

    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ZipHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DirectoryHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DatasetPreparer')
    def test_get_training_status(self, mock_preparer, mock_dir, mock_zip, mock_config_loader):
        """Test getting training status."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.create_system_constants.return_value = SystemConstants()
        mock_loader_instance.create_experiment_config.return_value = ExperimentConfig()
        mock_config_loader.return_value = mock_loader_instance

        trainer = FederatedTrainer()
        status = trainer.get_training_status()

        assert isinstance(status, dict)
        assert 'checkpoint_dir' in status
        assert 'logs_dir' in status
        assert 'partition_strategy' in status
        assert 'config' in status

    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ZipHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DirectoryHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DatasetPreparer')
    def test_validate_source_nonexistent(self, mock_preparer, mock_dir, mock_zip, mock_config_loader):
        """Test validation of nonexistent source."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.create_system_constants.return_value = SystemConstants()
        mock_loader_instance.create_experiment_config.return_value = ExperimentConfig()
        mock_config_loader.return_value = mock_loader_instance

        trainer = FederatedTrainer()
        result = trainer.validate_source("/nonexistent/path")

        assert isinstance(result, dict)
        assert 'valid' in result
        assert result['valid'] is False


