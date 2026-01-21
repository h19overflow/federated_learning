"""
Shared fixtures for federated learning tests.
"""

from typing import Dict, Generator
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid


@pytest.fixture
def sample_metadata_df() -> pd.DataFrame:
    """Create sample metadata DataFrame for testing."""
    return pd.DataFrame(
        {
            "patientId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "filename": [
                "1.png",
                "2.png",
                "3.png",
                "4.png",
                "5.png",
                "6.png",
                "7.png",
                "8.png",
                "9.png",
                "10.png",
            ],
            "class": [
                "Normal",
                "Pneumonia",
                "Normal",
                "Pneumonia",
                "Normal",
                "Pneumonia",
                "Normal",
                "Pneumonia",
                "Normal",
                "Pneumonia",
            ],
        },
    )


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """Create sample training DataFrame with patientId."""
    return pd.DataFrame(
        {
            "patientId": [1, 2, 3, 4, 5, 6, 7, 8],
            "class": [
                "Normal",
                "Pneumonia",
                "Normal",
                "Pneumonia",
                "Normal",
                "Pneumonia",
                "Normal",
                "Pneumonia",
            ],
        },
    )


@pytest.fixture
def sample_val_df() -> pd.DataFrame:
    """Create sample validation DataFrame with patientId."""
    return pd.DataFrame(
        {
            "patientId": [9, 10],
            "class": ["Normal", "Pneumonia"],
        },
    )


@pytest.fixture
def sample_pyproject_toml_content() -> str:
    """Sample pyproject.toml content for testing."""
    return """[project]
name = "test"

[tool.flwr.app]
config = {num-server-rounds = 5, max-epochs = 2}

[tool.flwr.federations.local-simulation]
options = {num-supernodes = 3}
"""


@pytest.fixture
def sample_train_config() -> Dict:
    """Sample training configuration."""
    return {
        "file_path": "test_data.csv",
        "image_dir": "test_images",
        "num_partitions": 2,
        "seed": 42,
        "run_id": 1,
    }


@pytest.fixture
def sample_eval_config() -> Dict:
    """Sample evaluation configuration."""
    return {
        "csv_path": "test_data.csv",
        "image_dir": "test_images",
    }


@pytest.fixture
def mock_pytorch_model() -> Mock:
    """Create mock PyTorch model."""
    model = Mock()
    model.state_dict.return_value = {
        "layer1.weight": torch.randn(64, 3, 7, 7),
        "layer1.bias": torch.randn(64),
        "layer2.weight": torch.randn(64),
    }
    return model


@pytest.fixture
def mock_array_record(mock_pytorch_model) -> ArrayRecord:
    """Create mock ArrayRecord with model state."""
    array_record = Mock(spec=ArrayRecord)
    array_record.to_torch_state_dict.return_value = mock_pytorch_model.state_dict()
    return array_record


@pytest.fixture
def mock_context() -> Mock:
    """Create mock Flower Context."""
    context = Mock(spec=Context)
    context.node_id = 0
    context.state = Mock()
    context.state.current_round = 1
    context.run_id = 1
    context.run_config = {"num-server-rounds": 3}
    return context


@pytest.fixture
def mock_grid() -> Mock:
    """Create mock Flower Grid."""
    grid = Mock(spec=Grid)
    grid.get_node_ids.return_value = iter([0, 1, 2])
    return grid


@pytest.fixture
def mock_config_record() -> Mock:
    """Create mock ConfigRecord."""
    config = Mock(spec=ConfigRecord)
    config.__getitem__ = lambda self, key: {"file_path": "test.csv"}.get(key)
    config.update = Mock()
    return config


@pytest.fixture
def mock_metric_record() -> MetricRecord:
    """Create mock MetricRecord."""
    return MetricRecord(
        {
            "test_loss": 0.5,
            "test_accuracy": 0.8,
            "test_precision": 0.75,
            "test_recall": 0.7,
            "test_f1": 0.72,
            "test_auroc": 0.85,
            "num-examples": 50,
        },
    )


@pytest.fixture
def mock_message(mock_array_record) -> Mock:
    """Create mock Flower Message."""
    msg = Mock(spec=Message)
    msg.content = RecordDict(
        {
            "arrays": mock_array_record,
            "config": {
                "file_path": "test.csv",
                "image_dir": "test_images",
                "num_partitions": 2,
                "seed": 42,
                "run_id": 1,
            },
        },
    )
    msg.reply_to = None
    return msg


@pytest.fixture
def mock_eval_message(mock_array_record) -> Mock:
    """Create mock evaluation Message."""
    msg = Mock(spec=Message)
    msg.content = RecordDict(
        {
            "arrays": mock_array_record,
            "config": {
                "csv_path": "test.csv",
                "image_dir": "test_images",
            },
        },
    )
    msg.reply_to = None
    return msg


@pytest.fixture
def mock_trainer() -> Mock:
    """Create mock Lightning trainer."""
    trainer = Mock()
    trainer.fit.return_value = None
    trainer.test.return_value = [
        {
            "test_loss": 0.5,
            "test_acc": 0.8,
            "test_precision": 0.75,
            "test_recall": 0.7,
            "test_f1": 0.72,
            "test_auroc": 0.85,
        },
    ]
    return trainer


@pytest.fixture
def mock_centralized_trainer(mock_trainer) -> Mock:
    """Create mock CentralizedTrainer."""
    trainer = Mock()
    trainer.config = Mock()
    trainer.logger = Mock()
    trainer._prepare_dataset = Mock(return_value=(pd.DataFrame(), pd.DataFrame()))
    trainer._create_data_module = Mock()
    trainer._build_model_and_callbacks = Mock(return_value=(Mock(), [], Mock()))
    trainer._build_trainer = Mock(return_value=mock_trainer)
    trainer._collect_training_results = Mock(
        return_value={
            "metrics_history": [
                {"epoch": 1, "train_loss": 0.5, "val_acc": 0.8},
            ],
        },
    )
    return trainer


@pytest.fixture
def mock_config_manager() -> Mock:
    """Create mock ConfigManager."""
    config = Mock()
    config.get = Mock(
        side_effect=lambda key, default=None: {
            "experiment.file-path": "test.csv",
            "experiment.image-dir": "test_images",
            "experiment.num-server-rounds": 3,
            "experiment.max-epochs": 2,
            "experiment.options.num-supernodes": 3,
            "experiment.seed": 42,
        }.get(key, default),
    )
    config.has_key = Mock(
        side_effect=lambda key: key
        in [
            "experiment.file-path",
            "experiment.image-dir",
            "experiment.num-server-rounds",
        ],
    )
    return config


@pytest.fixture
def mock_websocket_sender() -> Mock:
    """Create mock MetricsWebSocketSender."""
    sender = Mock()
    sender.send_training_mode = Mock()
    sender.send_round_metrics = Mock()
    sender.send_metrics = Mock()
    return sender


@pytest.fixture
def sample_aggregated_metrics() -> Dict:
    """Sample aggregated metrics from server."""
    return {
        "loss": 0.45,
        "test_loss": 0.45,
        "test_acc": 0.82,
        "val_acc": 0.82,
        "test_precision": 0.78,
        "val_precision": 0.78,
        "test_recall": 0.75,
        "val_recall": 0.75,
        "test_f1": 0.76,
        "val_f1": 0.76,
        "test_auroc": 0.88,
        "val_auroc": 0.88,
    }


@pytest.fixture
def sample_server_eval_metrics() -> Dict:
    """Sample server evaluation metrics."""
    return {
        "server_loss": 0.4,
        "server_accuracy": 0.85,
        "server_precision": 0.8,
        "server_recall": 0.78,
        "server_f1": 0.79,
        "server_auroc": 0.9,
        "server_cm_tp": 42.0,
        "server_cm_tn": 38.0,
        "server_cm_fp": 5.0,
        "server_cm_fn": 8.0,
    }


@pytest.fixture
def temp_csv_file(tmp_path, sample_metadata_df) -> Generator[str, None, None]:
    """Create temporary CSV file with sample data."""
    csv_path = tmp_path / "test_metadata.csv"
    sample_metadata_df.to_csv(csv_path, index=False)
    yield str(csv_path)


def create_mock_reply_message(metrics: Dict, num_examples: int) -> Mock:
    """Helper to create mock reply message."""
    msg = Mock(spec=Message)
    msg.content = RecordDict(
        {
            "metrics": MetricRecord({**metrics, "num-examples": num_examples}),
        },
    )
    return msg
