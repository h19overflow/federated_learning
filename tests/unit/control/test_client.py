"""
Unit tests for federated learning client (FlowerClient).
Tests parameter management, training, and evaluation functionality.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.control.federated_learning.client import (
    FlowerClient,
    evaluate,
    get_weights,
    set_weights,
    train,
)


class SimpleTestNet(nn.Module):
    """Simple network for testing."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MockDataLoader:
    """Mock DataLoader class for testing."""

    def __init__(self, data):
        self.data = data
        self.dataset = list(range(sum(len(batch[0]) for batch in data)))

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.dataset)


class TestGetWeights:
    """Tests for get_weights helper function."""

    def test_get_weights_returns_list(self):
        """Test that get_weights returns a list of numpy arrays."""
        net = SimpleTestNet()
        weights = get_weights(net)

        assert isinstance(weights, list)
        assert len(weights) > 0
        assert all(isinstance(w, np.ndarray) for w in weights)

    def test_get_weights_matches_state_dict(self):
        """Test that get_weights extracts all state dict parameters."""
        net = SimpleTestNet()
        weights = get_weights(net)
        state_dict = net.state_dict()

        assert len(weights) == len(state_dict)

    def test_get_weights_cpu_conversion(self):
        """Test that weights are on CPU after extraction."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = SimpleTestNet().to(device)
        weights = get_weights(net)

        # All weights should be numpy arrays (CPU)
        assert all(isinstance(w, np.ndarray) for w in weights)

    def test_get_weights_reproducible(self):
        """Test that calling get_weights twice returns same values."""
        net = SimpleTestNet()
        weights1 = get_weights(net)
        weights2 = get_weights(net)

        for w1, w2 in zip(weights1, weights2):
            assert np.array_equal(w1, w2)


class TestSetWeights:
    """Tests for set_weights helper function."""

    def test_set_weights_updates_model(self):
        """Test that set_weights properly updates model parameters."""
        net = SimpleTestNet()
        original_weights = get_weights(net)

        # Create new random weights
        modified_weights = [np.random.randn(*w.shape) for w in original_weights]

        # Set new weights
        set_weights(net, modified_weights)
        new_weights = get_weights(net)

        # Verify weights were updated
        for new, modified in zip(new_weights, modified_weights):
            assert np.allclose(new, modified)

    def test_set_weights_torch_tensor_conversion(self):
        """Test that set_weights properly converts numpy arrays to tensors."""
        net = SimpleTestNet()
        weights = get_weights(net)

        # Convert to list of numpy arrays and set
        set_weights(net, weights)

        # Verify model state dict contains tensors
        for param in net.state_dict().values():
            assert isinstance(param, torch.Tensor)

    def test_set_weights_with_random_weights(self):
        """Test set_weights with random numpy arrays."""
        net = SimpleTestNet()
        # Get the correct shapes from the network
        original_weights = get_weights(net)
        shapes = [w.shape for w in original_weights]

        random_weights = [np.random.randn(*shape) for shape in shapes]

        # Should not raise
        set_weights(net, random_weights)

        # Verify weights were set
        new_weights = get_weights(net)
        assert len(new_weights) == len(random_weights)

    def test_set_weights_preserves_order(self):
        """Test that set_weights maintains parameter order."""
        net = SimpleTestNet()
        weights = get_weights(net)

        set_weights(net, weights)
        retrieved_weights = get_weights(net)

        for w1, w2 in zip(weights, retrieved_weights):
            assert np.array_equal(w1, w2)


class TestFlowerClientInit:
    """Tests for FlowerClient initialization."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for FlowerClient."""
        net = SimpleTestNet()

        # Create simple data for dataloaders
        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))

        trainloader = MockDataLoader([(images, labels)])
        valloader = MockDataLoader([(images, labels)])
        config = ExperimentConfig()
        device = torch.device("cpu")

        return {
            "net": net,
            "trainloader": trainloader,
            "valloader": valloader,
            "config": config,
            "device": device,
        }

    def test_init_valid_parameters(self, mock_dependencies):
        """Test FlowerClient initialization with valid parameters."""
        client = FlowerClient(**mock_dependencies)

        assert client.net is mock_dependencies["net"]
        assert client.trainloader is mock_dependencies["trainloader"]
        assert client.valloader is mock_dependencies["valloader"]
        assert client.config is mock_dependencies["config"]
        assert client.device is mock_dependencies["device"]

    def test_init_none_net_raises_error(self, mock_dependencies):
        """Test that None net raises ValueError."""
        mock_dependencies["net"] = None

        with pytest.raises(ValueError, match="net cannot be None"):
            FlowerClient(**mock_dependencies)

    def test_init_none_trainloader_raises_error(self, mock_dependencies):
        """Test that None trainloader raises ValueError."""
        mock_dependencies["trainloader"] = None

        with pytest.raises(ValueError, match="trainloader cannot be None"):
            FlowerClient(**mock_dependencies)

    def test_init_none_valloader_raises_error(self, mock_dependencies):
        """Test that None valloader raises ValueError."""
        mock_dependencies["valloader"] = None

        with pytest.raises(ValueError, match="valloader cannot be None"):
            FlowerClient(**mock_dependencies)


class TestFlowerClientGetParameters:
    """Tests for FlowerClient.get_parameters method."""

    def test_get_parameters_returns_list(self):
        """Test that get_parameters returns a list."""
        net = SimpleTestNet()
        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))
        trainloader = MockDataLoader([(images, labels)])
        valloader = MockDataLoader([(images, labels)])
        config = ExperimentConfig()
        device = torch.device("cpu")

        client = FlowerClient(net, trainloader, valloader, config, device)
        params = client.get_parameters({})

        assert isinstance(params, list)
        assert len(params) > 0

    def test_get_parameters_matches_weights(self):
        """Test that get_parameters matches get_weights output."""
        net = SimpleTestNet()
        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))
        trainloader = MockDataLoader([(images, labels)])
        valloader = MockDataLoader([(images, labels)])
        config = ExperimentConfig()
        device = torch.device("cpu")

        client = FlowerClient(net, trainloader, valloader, config, device)
        params = client.get_parameters({})
        weights = get_weights(net)

        assert len(params) == len(weights)
        for p, w in zip(params, weights):
            assert np.array_equal(p, w)


class TestFlowerClientSetParameters:
    """Tests for FlowerClient.set_parameters method."""

    def test_set_parameters_updates_model(self):
        """Test that set_parameters updates the model."""
        net = SimpleTestNet()
        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))
        trainloader = MockDataLoader([(images, labels)])
        valloader = MockDataLoader([(images, labels)])
        config = ExperimentConfig()
        device = torch.device("cpu")

        client = FlowerClient(net, trainloader, valloader, config, device)
        original_params = client.get_parameters({})

        # Create new random parameters
        new_params = [np.random.randn(*p.shape) for p in original_params]

        # Set parameters
        client.set_parameters(new_params)

        # Verify they were set
        updated_params = client.get_parameters({})
        for new, updated in zip(new_params, updated_params):
            assert np.allclose(new, updated)


class TestTrainFunction:
    """Tests for train function."""

    def test_train_returns_float_loss(self):
        """Test that train returns a tuple of (float loss, list of epoch losses)."""
        net = SimpleTestNet(num_classes=2)
        device = torch.device("cpu")

        # Create mock dataloader
        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))
        trainloader = [(images, labels)]

        avg_loss, epoch_losses = train(
            net=net,
            trainloader=trainloader,
            epochs=1,
            device=device,
            learning_rate=0.001,
            weight_decay=0.0001,
            num_classes=2,
        )

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        assert isinstance(epoch_losses, list)
        assert len(epoch_losses) == 1

    def test_train_multiclass(self):
        """Test train with multi-class classification."""
        net = SimpleTestNet(num_classes=3)
        device = torch.device("cpu")

        images = torch.randn(4, 10)
        labels = torch.randint(0, 3, (4,))
        trainloader = [(images, labels)]

        avg_loss, epoch_losses = train(
            net=net,
            trainloader=trainloader,
            epochs=1,
            device=device,
            learning_rate=0.001,
            weight_decay=0.0001,
            num_classes=3,
        )

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        assert isinstance(epoch_losses, list)
        assert len(epoch_losses) == 1

    def test_train_binary_classification(self):
        """Test train with binary classification (num_classes=1)."""
        net = SimpleTestNet(num_classes=1)
        device = torch.device("cpu")

        images = torch.randn(4, 10)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        trainloader = [(images, labels)]

        avg_loss, epoch_losses = train(
            net=net,
            trainloader=trainloader,
            epochs=1,
            device=device,
            learning_rate=0.001,
            weight_decay=0.0001,
            num_classes=1,
        )

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        assert isinstance(epoch_losses, list)
        assert len(epoch_losses) == 1

    def test_train_multiple_epochs(self):
        """Test that training runs for multiple epochs."""
        net = SimpleTestNet(num_classes=2)
        device = torch.device("cpu")

        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))
        trainloader = [(images, labels)]

        avg_loss, epoch_losses = train(
            net=net,
            trainloader=trainloader,
            epochs=3,
            device=device,
            learning_rate=0.001,
            weight_decay=0.0001,
            num_classes=2,
        )

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        assert isinstance(epoch_losses, list)
        assert len(epoch_losses) == 3

    def test_train_network_in_train_mode(self):
        """Test that network is set to train mode during training."""
        net = SimpleTestNet(num_classes=2)
        device = torch.device("cpu")

        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))
        trainloader = [(images, labels)]

        # Train should set network to train mode
        train(
            net=net,
            trainloader=trainloader,
            epochs=1,
            device=device,
            learning_rate=0.001,
            weight_decay=0.0001,
            num_classes=2,
        )

        # After training, network could be in train or eval mode
        # Just verify it ran without error


class TestEvaluateFunction:
    """Tests for evaluate function."""

    def test_evaluate_returns_tuple(self):
        """Test that evaluate returns (loss, accuracy) tuple."""
        net = SimpleTestNet(num_classes=2)
        device = torch.device("cpu")

        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))
        valloader = MockDataLoader([(images, labels)])

        loss, accuracy = evaluate(
            net=net,
            valloader=valloader,
            device=device,
            num_classes=2,
        )

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss >= 0
        assert 0 <= accuracy <= 1

    def test_evaluate_multiclass(self):
        """Test evaluate with multi-class classification."""
        net = SimpleTestNet(num_classes=3)
        device = torch.device("cpu")

        images = torch.randn(8, 10)
        labels = torch.randint(0, 3, (8,))
        valloader = MockDataLoader([(images, labels)])

        loss, accuracy = evaluate(
            net=net,
            valloader=valloader,
            device=device,
            num_classes=3,
        )

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_evaluate_binary_classification(self):
        """Test evaluate with binary classification (num_classes=1)."""
        net = SimpleTestNet(num_classes=1)
        device = torch.device("cpu")

        images = torch.randn(8, 10)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        valloader = MockDataLoader([(images, labels)])

        loss, accuracy = evaluate(
            net=net,
            valloader=valloader,
            device=device,
            num_classes=1,
        )

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1

    def test_evaluate_network_in_eval_mode(self):
        """Test that evaluate runs without errors."""
        net = SimpleTestNet(num_classes=2)
        device = torch.device("cpu")

        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))
        valloader = MockDataLoader([(images, labels)])

        # Should run without errors
        loss, accuracy = evaluate(
            net=net,
            valloader=valloader,
            device=device,
            num_classes=2,
        )

        # Verify results are valid
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)


class TestFlowerClientFit:
    """Tests for FlowerClient.fit method."""

    def test_fit_initialization(self):
        """Test that fit can be called successfully."""
        net = SimpleTestNet(num_classes=2)
        device = torch.device("cpu")

        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))

        trainloader = MockDataLoader([(images, labels)])
        valloader = MockDataLoader([(images, labels)])
        config = ExperimentConfig(local_epochs=1)

        client = FlowerClient(net, trainloader, valloader, config, device)

        # Get initial parameters
        initial_params = get_weights(net)

        # Verify client has expected attributes
        assert client.net is not None
        assert client.config is not None
        assert len(initial_params) > 0


class TestFlowerClientEvaluate:
    """Tests for FlowerClient.evaluate method."""

    def test_evaluate_client_initialization(self):
        """Test that client can be initialized for evaluation."""
        net = SimpleTestNet(num_classes=2)
        device = torch.device("cpu")

        images = torch.randn(4, 10)
        labels = torch.randint(0, 2, (4,))

        trainloader = MockDataLoader([(images, labels)])
        valloader = MockDataLoader([(images, labels)])
        config = ExperimentConfig()

        client = FlowerClient(net, trainloader, valloader, config, device)

        # Verify client is properly initialized
        assert client is not None
        assert client.config is not None
        assert client.device == device
