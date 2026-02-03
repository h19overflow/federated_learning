"""
Pytest fixtures specific to experiment endpoints testing.

Provides fixtures for mocking trainers, Flower components, file uploads,
and experiment tracking.
"""

import io
import json
import os
import tempfile
import zipfile
from typing import Dict
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile

from federated_pneumonia_detection.src.internals.loggers.logger import get_logger


@pytest.fixture
def mock_upload_file() -> UploadFile:
    """Create mock UploadFile object for testing."""
    file = UploadFile(filename="test_data.zip")
    file.file = io.BytesIO()
    return file


@pytest.fixture
def mock_centralized_trainer():
    """Create mock CentralizedTrainer."""
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = {
        "final_metrics": {
            "train_loss": 0.5,
            "val_loss": 0.4,
            "accuracy": 0.85,
            "f1_score": 0.83,
        },
        "checkpoint_path": "test_checkpoint.ckpt",
    }
    return mock_trainer


@pytest.fixture
def mock_pytorch_lightning_trainer():
    """Create mock PyTorch Lightning Trainer."""
    mock_trainer = MagicMock()
    mock_trainer.fit = MagicMock()
    mock_trainer.test = MagicMock()
    mock_trainer.validate = MagicMock()
    return mock_trainer


@pytest.fixture
def mock_flower_server():
    """Create mock Flower server."""
    mock_server = MagicMock()
    mock_server.start = MagicMock()
    mock_server.start.return_value = 0
    return mock_server


@pytest.fixture
def mock_config_manager():
    """Create mock ConfigManager."""
    mock_config = MagicMock()
    mock_config.get = MagicMock(return_value="test_value")
    mock_config.set = MagicMock()
    mock_config.save = MagicMock()
    return mock_config


@pytest.fixture
def sample_zip_file(tmp_path) -> str:
    """Create sample ZIP file with Images/ and metadata CSV."""
    zip_path = tmp_path / "test_data.zip"

    # Create a temporary directory with test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create Images directory
        images_dir = os.path.join(temp_dir, "Images")
        os.makedirs(images_dir)

        # Create sample image files
        for i in range(3):
            img_path = os.path.join(images_dir, f"image_{i}.jpg")
            with open(img_path, "wb") as f:
                f.write(b"fake_image_data")

        # Create metadata CSV
        csv_path = os.path.join(temp_dir, "stage2_train_metadata.csv")
        with open(csv_path, "w") as f:
            f.write("file_name,label\n")
            for i in range(3):
                f.write(f"image_{i}.jpg,{i % 2}\n")

        # Create ZIP
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

    return str(zip_path)


@pytest.fixture
def sample_zip_with_root_dir(tmp_path) -> str:
    """Create ZIP file with root directory wrapper."""
    zip_path = tmp_path / "test_data_wrapped.zip"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create root directory
        root_dir = os.path.join(temp_dir, "dataset_root")

        # Create Images directory
        images_dir = os.path.join(root_dir, "Images")
        os.makedirs(images_dir)

        # Create sample image files
        for i in range(2):
            img_path = os.path.join(images_dir, f"img_{i}.jpg")
            with open(img_path, "wb") as f:
                f.write(b"fake_img_data")

        # Create metadata CSV
        csv_path = os.path.join(root_dir, "metadata.csv")
        with open(csv_path, "w") as f:
            f.write("file_name,label\nimg_0.jpg,0\nimg_1.jpg,1\n")

        # Create ZIP
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

    return str(zip_path)


@pytest.fixture
def sample_experiment_log(tmp_path) -> Dict:
    """Create sample experiment log JSON."""
    log_data = {
        "metadata": {
            "experiment_name": "test_experiment",
            "status": "running",
            "training_mode": "centralized",
            "start_time": "2025-01-21T10:00:00",
            "end_time": None,
        },
        "current_epoch": 3,
        "epochs": [
            {
                "type": "epoch_start",
                "epoch": 1,
                "total_epochs": 10,
                "timestamp": "2025-01-21T10:00:10",
            },
            {
                "type": "epoch_end",
                "epoch": 1,
                "metrics": {
                    "train_loss": 0.8,
                    "val_loss": 0.7,
                    "accuracy": 0.65,
                },
                "timestamp": "2025-01-21T10:05:00",
            },
            {
                "type": "epoch_start",
                "epoch": 2,
                "total_epochs": 10,
                "timestamp": "2025-01-21T10:05:10",
            },
            {
                "type": "epoch_end",
                "epoch": 2,
                "metrics": {
                    "train_loss": 0.6,
                    "val_loss": 0.5,
                    "accuracy": 0.75,
                },
                "timestamp": "2025-01-21T10:10:00",
            },
            {
                "type": "epoch_start",
                "epoch": 3,
                "total_epochs": 10,
                "timestamp": "2025-01-21T10:10:10",
            },
            {
                "type": "epoch_end",
                "epoch": 3,
                "metrics": {
                    "train_loss": 0.5,
                    "val_loss": 0.4,
                    "accuracy": 0.82,
                },
                "timestamp": "2025-01-21T10:15:00",
            },
        ],
    }

    log_file = tmp_path / "test_experiment.json"
    with open(log_file, "w") as f:
        json.dump(log_data, f)

    return log_data


@pytest.fixture
def sample_completed_experiment_log(tmp_path) -> Dict:
    """Create sample completed experiment log."""
    log_data = {
        "metadata": {
            "experiment_name": "completed_experiment",
            "status": "completed",
            "training_mode": "centralized",
            "start_time": "2025-01-21T09:00:00",
            "end_time": "2025-01-21T10:00:00",
        },
        "current_epoch": 10,
        "epochs": [
            {
                "type": "epoch_start",
                "epoch": 1,
                "total_epochs": 10,
                "timestamp": "2025-01-21T09:00:10",
            },
            {
                "type": "epoch_end",
                "epoch": 1,
                "metrics": {"train_loss": 0.8, "val_loss": 0.7, "accuracy": 0.65},
                "timestamp": "2025-01-21T09:10:00",
            },
        ]
        * 10,  # 10 epochs
    }

    log_file = tmp_path / "completed_experiment.json"
    with open(log_file, "w") as f:
        json.dump(log_data, f)

    return log_data


@pytest.fixture
def experiment_logger():
    """Create logger instance for experiment tests."""
    return get_logger("test_experiment")


@pytest.fixture
def mock_background_tasks():
    """Create mock BackgroundTasks object."""
    mock_tasks = MagicMock()
    mock_tasks.add_task = MagicMock()
    return mock_tasks


@pytest.fixture
def temp_experiment_dirs(tmp_path) -> Dict[str, str]:
    """Create temporary directories for experiment checkpoints and logs."""
    checkpoint_dir = tmp_path / "checkpoints"
    logs_dir = tmp_path / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "logs_dir": str(logs_dir),
    }


@pytest.fixture
def invalid_zip_file(tmp_path) -> str:
    """Create invalid ZIP file for error testing."""
    zip_path = tmp_path / "invalid.zip"
    with open(zip_path, "wb") as f:
        f.write(b"not_a_zip_file")
    return str(zip_path)


@pytest.fixture
def zip_missing_images(tmp_path) -> str:
    """Create ZIP file without Images directory."""
    zip_path = tmp_path / "no_images.zip"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create only CSV (no Images)
        csv_path = os.path.join(temp_dir, "metadata.csv")
        with open(csv_path, "w") as f:
            f.write("file_name,label\nimg.jpg,0\n")

        # Create ZIP
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

    return str(zip_path)


@pytest.fixture
def mock_subprocess_popen():
    """Create mock for subprocess.Popen."""
    mock_process = MagicMock()
    mock_process.pid = 12345
    mock_process.wait.return_value = 0
    mock_process.stdout = io.StringIO(
        "Flower training started...\nRound 1 complete...\nTraining complete\n",
    )
    return mock_process
