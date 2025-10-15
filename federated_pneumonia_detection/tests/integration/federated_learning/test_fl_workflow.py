"""
Integration tests for complete federated learning workflows.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

from federated_pneumonia_detection.src.control.federated_learning.data_partitioner import (
    partition_data_iid,
    partition_data_stratified
)
from federated_pneumonia_detection.src.control.federated_learning.training_functions import (
    get_model_parameters,
    set_model_parameters
)
from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig


@pytest.mark.integration
@pytest.mark.federated
class TestEndToEndDataPartitioning:
    """Test complete data partitioning workflow."""

    def test_partition_and_verify_integrity(self):
        """Test that data partitioning maintains data integrity."""
        # Create test dataset
        np.random.seed(42)
        num_samples = 100
        df = pd.DataFrame({
            'patientId': [f'patient_{i:04d}' for i in range(num_samples)],
            'Target': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]),
            'filename': [f'patient_{i:04d}.png' for i in range(num_samples)]
        })

        # Partition using IID
        partitions = partition_data_iid(df, num_clients=5, seed=42)

        # Verify integrity
        assert len(partitions) == 5

        # Check all data is preserved
        total_samples = sum(len(p) for p in partitions)
        assert total_samples == len(df)

        # Check no duplicate patients
        all_patients = []
        for partition in partitions:
            all_patients.extend(partition['patientId'].tolist())

        assert len(all_patients) == len(set(all_patients))
        assert set(all_patients) == set(df['patientId'])

    def test_stratified_maintains_class_balance(self):
        """Test that stratified partitioning maintains class distribution."""
        # Create imbalanced dataset
        np.random.seed(42)
        num_samples = 120
        df = pd.DataFrame({
            'patientId': [f'patient_{i:04d}' for i in range(num_samples)],
            'Target': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
            'filename': [f'patient_{i:04d}.png' for i in range(num_samples)]
        })

        # Get original class distribution
        original_positive_ratio = (df['Target'] == 1).sum() / len(df)

        # Partition with stratification
        partitions = partition_data_stratified(df, num_clients=4, target_column='Target', seed=42)

        # Check each partition has similar class distribution
        for partition in partitions:
            partition_positive_ratio = (partition['Target'] == 1).sum() / len(partition)
            # Should be within 0.15 of original ratio
            assert abs(partition_positive_ratio - original_positive_ratio) < 0.15


@pytest.mark.integration
@pytest.mark.federated
class TestModelParameterExchange:
    """Test parameter exchange between models (simulating federated updates)."""

    def test_parameter_transfer_between_models(self):
        """Test transferring parameters from one model to another."""
        import torch
        import torch.nn as nn

        # Create two identical models
        model1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        model2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

        # Train model1 slightly (simulate local training)
        optimizer = torch.optim.SGD(model1.parameters(), lr=0.01)
        dummy_input = torch.randn(32, 10)
        dummy_target = torch.randn(32, 1)

        for _ in range(5):
            output = model1(dummy_input)
            loss = nn.MSELoss()(output, dummy_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Transfer parameters from model1 to model2
        params = get_model_parameters(model1)
        set_model_parameters(model2, params)

        # Verify parameters match
        params1_after = get_model_parameters(model1)
        params2_after = get_model_parameters(model2)

        for p1, p2 in zip(params1_after, params2_after):
            assert np.allclose(p1, p2)

    def test_federated_averaging_simulation(self):
        """Test simple federated averaging of model parameters."""
        import torch
        import torch.nn as nn

        # Create 3 client models
        models = [
            nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
            for _ in range(3)
        ]

        # Get parameters from all clients
        client_params = [get_model_parameters(m) for m in models]

        # Simulate FedAvg: average parameters
        num_clients = len(client_params)
        num_layers = len(client_params[0])

        averaged_params = []
        for layer_idx in range(num_layers):
            layer_params = [client_params[i][layer_idx] for i in range(num_clients)]
            avg_param = np.mean(layer_params, axis=0)
            averaged_params.append(avg_param)

        # Create global model and set averaged parameters
        global_model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
        set_model_parameters(global_model, averaged_params)

        # Verify global model has averaged parameters
        global_params = get_model_parameters(global_model)
        for gp, ap in zip(global_params, averaged_params):
            assert np.allclose(gp, ap)


@pytest.mark.integration
class TestFederatedTrainerWorkflow:
    """Test FederatedTrainer end-to-end workflow."""

    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ZipHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DirectoryHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DatasetPreparer')
    def test_trainer_initialization_and_status(self, mock_preparer, mock_dir, mock_zip, mock_config_loader):
        """Test complete trainer initialization and status retrieval."""
        from federated_pneumonia_detection.src.control.federated_learning.federated_trainer import FederatedTrainer

        # Setup mocks with real config objects
        mock_loader_instance = MagicMock()
        mock_loader_instance.create_system_constants.return_value = SystemConstants()
        mock_loader_instance.create_experiment_config.return_value = ExperimentConfig()
        mock_config_loader.return_value = mock_loader_instance

        # Create trainer
        trainer = FederatedTrainer(
            checkpoint_dir="test_checkpoints",
            logs_dir="test_logs",
            partition_strategy="iid"
        )

        # Get status
        status = trainer.get_training_status()

        # Verify status contains expected fields
        assert 'checkpoint_dir' in status
        assert 'logs_dir' in status
        assert 'partition_strategy' in status
        assert status['partition_strategy'] == 'iid'

    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ConfigLoader')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.ZipHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DirectoryHandler')
    @patch('federated_pneumonia_detection.src.control.federated_learning.federated_trainer.DatasetPreparer')
    def test_trainer_partition_workflow(self, mock_preparer, mock_dir, mock_zip, mock_config_loader):
        """Test data partitioning through trainer."""
        from federated_pneumonia_detection.src.control.federated_learning.federated_trainer import FederatedTrainer

        # Setup mocks with real config objects
        mock_loader_instance = MagicMock()
        mock_loader_instance.create_system_constants.return_value = SystemConstants()
        mock_loader_instance.create_experiment_config.return_value = ExperimentConfig(num_clients=3)
        mock_config_loader.return_value = mock_loader_instance

        # Create test data
        test_df = pd.DataFrame({
            'patientId': [f'p{i}' for i in range(60)],
            'Target': [0, 1] * 30,
            'filename': [f'p{i}.png' for i in range(60)]
        })

        # Create trainer
        trainer = FederatedTrainer(partition_strategy="stratified")

        # Partition data
        partitions = trainer._partition_data_for_clients(test_df)

        # Verify partitions
        assert len(partitions) > 0
        total_samples = sum(len(p) for p in partitions)
        assert total_samples == len(test_df)



