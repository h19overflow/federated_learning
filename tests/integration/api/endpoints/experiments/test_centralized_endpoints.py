"""
Integration tests for centralized experiment endpoints.

Tests cover:
- POST /experiments/centralized/train
- File upload handling
- Background task scheduling
- Error handling
- Response validation
"""

from unittest.mock import patch

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from federated_pneumonia_detection.src.api.endpoints.experiments import (
    centralized_endpoints,
)


@pytest.fixture
def client():
    """Create test client for centralized endpoints."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(centralized_endpoints.router)
    return TestClient(app)


@pytest.fixture
def mock_background_tasks():
    """Create mock BackgroundTasks."""
    return BackgroundTasks()


class TestCentralizedTrainEndpoint:
    """Test POST /experiments/centralized/train endpoint."""

    def test_start_centralized_training_success(
        self,
        client,
        sample_zip_file,
        mock_centralized_trainer,
    ):
        """Test successful start of centralized training."""
        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.centralized_endpoints.prepare_zip",
                return_value="/tmp/extracted_data",
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
                return_value=mock_centralized_trainer,
            ),
        ):
            with open(sample_zip_file, "rb") as f:
                response = client.post(
                    "/experiments/centralized/train",
                    files={"data_zip": ("test.zip", f, "application/zip")},
                    data={
                        "checkpoint_dir": "test_checkpoints",
                        "logs_dir": "test_logs",
                        "experiment_name": "test_exp",
                        "csv_filename": "test.csv",
                    },
                )

            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Centralized training started successfully"
            assert data["experiment_name"] == "test_exp"
            assert data["status"] == "queued"
            assert data["checkpoint_dir"] == "test_checkpoints"
            assert data["logs_dir"] == "test_logs"

    def test_start_centralized_training_default_params(
        self,
        client,
        sample_zip_file,
        mock_centralized_trainer,
    ):
        """Test centralized training with default parameters."""
        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.centralized_endpoints.prepare_zip",
                return_value="/tmp/extracted_data",
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
                return_value=mock_centralized_trainer,
            ),
        ):
            with open(sample_zip_file, "rb") as f:
                response = client.post(
                    "/experiments/centralized/train",
                    files={"data_zip": ("test.zip", f, "application/zip")},
                )

            assert response.status_code == 200
            data = response.json()
            assert data["experiment_name"] == "pneumonia_centralized"
            assert data["checkpoint_dir"] == "results/centralized/checkpoints"
            assert data["logs_dir"] == "results/centralized/logs"

    def test_start_centralized_training_invalid_zip(
        self,
        client,
        invalid_zip_file,
    ):
        """Test error handling for invalid ZIP file."""
        with pytest.raises(Exception):
            with open(invalid_zip_file, "rb") as f:
                client.post(
                    "/experiments/centralized/train",
                    files={"data_zip": ("invalid.zip", f, "application/zip")},
                )

    def test_start_centralized_training_missing_file(
        self,
        client,
    ):
        """Test error when no file is uploaded."""
        response = client.post(
            "/experiments/centralized/train",
            files={},
        )
        # FastAPI should return 422 for missing required field
        assert response.status_code == 422

    def test_start_centralized_training_task_queued(
        self,
        client,
        sample_zip_file,
        mock_centralized_trainer,
    ):
        """Test that background task is queued properly."""
        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.centralized_endpoints.prepare_zip",
                return_value="/tmp/extracted_data",
            ) as mock_prepare,
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
                return_value=mock_centralized_trainer,
            ),
        ):
            with open(sample_zip_file, "rb") as f:
                response = client.post(
                    "/experiments/centralized/train",
                    files={"data_zip": ("test.zip", f, "application/zip")},
                    data={"experiment_name": "queue_test"},
                )

            # Verify prepare_zip was called
            assert response.status_code == 200
            mock_prepare.assert_called_once()

    def test_start_centralized_training_multiple_experiments(
        self,
        client,
        sample_zip_file,
        mock_centralized_trainer,
    ):
        """Test starting multiple concurrent experiments."""
        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.centralized_endpoints.prepare_zip",
                return_value="/tmp/extracted_data",
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
                return_value=mock_centralized_trainer,
            ),
        ):
            experiment_names = ["exp_001", "exp_002", "exp_003"]

            for exp_name in experiment_names:
                with open(sample_zip_file, "rb") as f:
                    response = client.post(
                        "/experiments/centralized/train",
                        files={"data_zip": ("test.zip", f, "application/zip")},
                        data={"experiment_name": exp_name},
                    )
                    assert response.status_code == 200
                    assert response.json()["experiment_name"] == exp_name

    def test_start_centralized_training_custom_directories(
        self,
        client,
        sample_zip_file,
        mock_centralized_trainer,
    ):
        """Test with custom checkpoint and log directories."""
        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.centralized_endpoints.prepare_zip",
                return_value="/tmp/extracted_data",
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
                return_value=mock_centralized_trainer,
            ),
        ):
            with open(sample_zip_file, "rb") as f:
                response = client.post(
                    "/experiments/centralized/train",
                    files={"data_zip": ("test.zip", f, "application/zip")},
                    data={
                        "checkpoint_dir": "/custom/checkpoints",
                        "logs_dir": "/custom/logs",
                        "experiment_name": "custom_dir_test",
                    },
                )

            assert response.status_code == 200
            data = response.json()
            assert data["checkpoint_dir"] == "/custom/checkpoints"
            assert data["logs_dir"] == "/custom/logs"

    def test_start_centralized_training_error_in_prepare_zip(
        self,
        client,
        sample_zip_file,
    ):
        """Test error handling when prepare_zip raises exception."""
        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.centralized_endpoints.prepare_zip",
            side_effect=RuntimeError("ZIP extraction failed"),
        ):
            with pytest.raises(RuntimeError):
                with open(sample_zip_file, "rb") as f:
                    client.post(
                        "/experiments/centralized/train",
                        files={"data_zip": ("test.zip", f, "application/zip")},
                    )

    def test_start_centralized_training_response_structure(
        self,
        client,
        sample_zip_file,
        mock_centralized_trainer,
    ):
        """Test that response contains all required fields."""
        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.centralized_endpoints.prepare_zip",
                return_value="/tmp/extracted_data",
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
                return_value=mock_centralized_trainer,
            ),
        ):
            with open(sample_zip_file, "rb") as f:
                response = client.post(
                    "/experiments/centralized/train",
                    files={"data_zip": ("test.zip", f, "application/zip")},
                )

            data = response.json()
            required_fields = [
                "message",
                "experiment_name",
                "checkpoint_dir",
                "logs_dir",
                "status",
            ]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"

    def test_start_centralized_training_different_csv_names(
        self,
        client,
        sample_zip_file,
        mock_centralized_trainer,
    ):
        """Test with different CSV filenames."""
        csv_names = [
            "metadata.csv",
            "train_data.csv",
            "custom_metadata.csv",
        ]

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.centralized_endpoints.prepare_zip",
                return_value="/tmp/extracted_data",
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
                return_value=mock_centralized_trainer,
            ),
        ):
            for csv_name in csv_names:
                with open(sample_zip_file, "rb") as f:
                    response = client.post(
                        "/experiments/centralized/train",
                        files={"data_zip": ("test.zip", f, "application/zip")},
                        data={
                            "csv_filename": csv_name,
                            "experiment_name": f"test_{csv_name}",
                        },
                    )

                assert response.status_code == 200

    def test_start_centralized_training_experiment_name_validation(
        self,
        client,
        sample_zip_file,
        mock_centralized_trainer,
    ):
        """Test with various experiment name formats."""
        experiment_names = [
            "simple",
            "with_underscores",
            "with-dashes",
            "CamelCase",
            "123_numbers",
            "with spaces",  # May need encoding
        ]

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.centralized_endpoints.prepare_zip",
                return_value="/tmp/extracted_data",
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.centralized_tasks.CentralizedTrainer",
                return_value=mock_centralized_trainer,
            ),
        ):
            for exp_name in experiment_names:
                with open(sample_zip_file, "rb") as f:
                    response = client.post(
                        "/experiments/centralized/train",
                        files={"data_zip": ("test.zip", f, "application/zip")},
                        data={"experiment_name": exp_name},
                    )

                assert response.status_code == 200
                assert exp_name in response.json()["experiment_name"]
