"""
Federated learning specific test fixtures.
"""

import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import Mock, MagicMock
from pathlib import Path
import tempfile

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig


@pytest.fixture
def fl_constants():
    """Create SystemConstants configured for federated learning tests."""
    return SystemConstants.create_custom(
        img_size=(224, 224),
        batch_size=8,
        sample_fraction=0.5,
        validation_split=0.2,
        seed=42,
        base_path='test_data',
        metadata_filename='test_metadata.csv'
    )


@pytest.fixture
def fl_config():
    """Create ExperimentConfig for federated learning tests."""
    return ExperimentConfig(
        learning_rate=0.001,
        epochs=2,
        batch_size=8,
        sample_fraction=0.5,
        validation_split=0.2,
        seed=42,
        num_rounds=3,
        num_clients=3,
        clients_per_round=2,
        local_epochs=2,
        weight_decay=0.0001
    )


@pytest.fixture
def federated_df():
    """Create DataFrame suitable for federated partitioning."""
    np.random.seed(42)
    num_samples = 60
    
    patient_ids = [f"patient_{i:04d}" for i in range(num_samples)]
    targets = np.random.choice([0, 1], num_samples, p=[0.6, 0.4])
    
    return pd.DataFrame({
        'patientId': patient_ids,
        'Target': targets,
        'filename': [f"{pid}.png" for pid in patient_ids]
    })


@pytest.fixture
def small_federated_df():
    """Create small DataFrame for quick tests."""
    return pd.DataFrame({
        'patientId': [f'pat_{i}' for i in range(12)],
        'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'filename': [f'pat_{i}.png' for i in range(12)]
    })


@pytest.fixture
def mock_torch_model():
    """Create mock PyTorch model."""
    model = MagicMock()
    model.train = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    model.parameters = Mock(return_value=[
        torch.randn(10, 10),
        torch.randn(10)
    ])
    model.state_dict = Mock(return_value={
        'layer1.weight': torch.randn(10, 10),
        'layer1.bias': torch.randn(10)
    })
    model.load_state_dict = Mock()
    return model


@pytest.fixture
def mock_dataloader():
    """Create mock DataLoader."""
    mock_loader = MagicMock()
    # Simulate batch iteration
    mock_loader.__iter__ = Mock(return_value=iter([
        (torch.randn(8, 3, 224, 224), torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]))
    ]))
    mock_loader.__len__ = Mock(return_value=1)
    return mock_loader


@pytest.fixture
def temp_fl_dirs():
    """Create temporary directories for FL tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        checkpoint_dir = temp_path / "checkpoints"
        logs_dir = temp_path / "logs"
        data_dir = temp_path / "data"
        images_dir = data_dir / "Images" / "Images"
        
        checkpoint_dir.mkdir()
        logs_dir.mkdir()
        images_dir.mkdir(parents=True)
        
        yield {
            'base': temp_path,
            'checkpoint': checkpoint_dir,
            'logs': logs_dir,
            'data': data_dir,
            'images': images_dir
        }


@pytest.fixture
def mock_config_loader():
    """Create mock ConfigLoader for FL tests."""
    mock_loader = Mock()
    
    constants = SystemConstants.create_custom(seed=42)
    config = ExperimentConfig(num_clients=3, num_rounds=2)
    
    mock_loader.load_config = Mock(return_value={})
    mock_loader.create_system_constants = Mock(return_value=constants)
    mock_loader.create_experiment_config = Mock(return_value=config)
    
    return mock_loader


@pytest.fixture
def partitioned_data(federated_df):
    """Create pre-partitioned data for testing."""
    from federated_pneumonia_detection.src.control.federated_learning.data_partitioner import partition_data_iid
    return partition_data_iid(federated_df, num_clients=3, seed=42)


