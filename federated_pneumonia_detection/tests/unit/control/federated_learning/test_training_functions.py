"""
Unit tests for PyTorch training functions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from federated_pneumonia_detection.src.control.federated_learning.training.functions import (
    train_one_epoch,
    evaluate_model,
    get_model_parameters,
    set_model_parameters,
    create_optimizer,
    train_multiple_epochs
)


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    return model


@pytest.fixture
def binary_dataloader():
    """Create simple dataloader for binary classification."""
    # Create simple dataset
    images = torch.randn(16, 3, 224, 224)
    labels = torch.randint(0, 2, (16,))
    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.mark.unit
class TestTrainingFunctions:
    """Test training loop functions."""

    def test_train_one_epoch_executes(self, simple_model, binary_dataloader):
        """Test that train_one_epoch runs without errors."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        device = torch.device('cpu')

        loss = train_one_epoch(
            model=simple_model,
            dataloader=binary_dataloader,
            optimizer=optimizer,
            device=device,
            num_classes=1
        )

        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_one_epoch_returns_valid_loss(self, simple_model, binary_dataloader):
        """Test that training returns reasonable loss value."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        device = torch.device('cpu')

        loss = train_one_epoch(
            simple_model, binary_dataloader, optimizer, device, num_classes=1
        )

        assert 0 <= loss < 100  # Loss should be in reasonable range

    def test_evaluate_model_executes(self, simple_model, binary_dataloader):
        """Test that evaluate_model runs without errors."""
        device = torch.device('cpu')

        loss, acc, metrics = evaluate_model(
            simple_model, binary_dataloader, device, num_classes=1
        )

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert isinstance(metrics, dict)
        assert 0 <= acc <= 1

    def test_evaluate_model_returns_metrics(self, simple_model, binary_dataloader):
        """Test that evaluation returns all expected metrics."""
        device = torch.device('cpu')

        _, _, metrics = evaluate_model(
            simple_model, binary_dataloader, device, num_classes=1
        )

        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'num_samples' in metrics
        assert metrics['num_samples'] > 0


@pytest.mark.unit
class TestParameterFunctions:
    """Test model parameter handling functions."""

    def test_get_model_parameters(self, simple_model):
        """Test extracting model parameters."""
        params = get_model_parameters(simple_model)

        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_set_model_parameters(self, simple_model):
        """Test setting model parameters."""
        # Get original parameters
        original_params = get_model_parameters(simple_model)

        # Create new parameters (slightly modified)
        new_params = [p + 0.1 for p in original_params]

        # Set new parameters
        set_model_parameters(simple_model, new_params)

        # Verify parameters changed
        updated_params = get_model_parameters(simple_model)
        
        for orig, updated in zip(original_params, updated_params):
            assert not np.allclose(orig, updated)

    def test_get_set_parameters_roundtrip(self, simple_model):
        """Test that get/set parameters is reversible."""
        original_params = get_model_parameters(simple_model)
        
        set_model_parameters(simple_model, original_params)
        
        recovered_params = get_model_parameters(simple_model)

        for orig, recovered in zip(original_params, recovered_params):
            assert np.allclose(orig, recovered)


@pytest.mark.unit
class TestOptimizerCreation:
    """Test optimizer creation functions."""

    def test_create_optimizer(self, simple_model):
        """Test optimizer creation."""
        optimizer = create_optimizer(simple_model, learning_rate=0.001)

        assert isinstance(optimizer, torch.optim.Optimizer)
        assert optimizer.param_groups[0]['lr'] == 0.001

    def test_create_optimizer_with_weight_decay(self, simple_model):
        """Test optimizer with weight decay."""
        optimizer = create_optimizer(
            simple_model, 
            learning_rate=0.01,
            weight_decay=0.0001
        )

        assert optimizer.param_groups[0]['weight_decay'] == 0.0001


@pytest.mark.unit
class TestMultiEpochTraining:
    """Test multi-epoch training function."""

    def test_train_multiple_epochs(self, simple_model, binary_dataloader):
        """Test training for multiple epochs."""
        device = torch.device('cpu')
        
        history = train_multiple_epochs(
            model=simple_model,
            train_loader=binary_dataloader,
            val_loader=binary_dataloader,
            num_epochs=2,
            learning_rate=0.001,
            device=device,
            num_classes=1
        )

        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'val_accuracy' in history
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 2

    def test_train_multiple_epochs_without_validation(self, simple_model, binary_dataloader):
        """Test training without validation loader."""
        device = torch.device('cpu')
        
        history = train_multiple_epochs(
            model=simple_model,
            train_loader=binary_dataloader,
            val_loader=None,
            num_epochs=2,
            learning_rate=0.001,
            device=device
        )

        assert len(history['train_loss']) == 2
        assert history['val_loss'] == []


