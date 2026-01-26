"""
Unit tests for federated training task functions.

Tests cover:
- Background task execution
- Configuration updates
- Subprocess spawning
- Environment variable handling
- Error handling for missing files
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks import (
    run_federated_training_task,
)


@pytest.fixture(autouse=True)
def mock_federated_modules():
    """
    Automatically mock federated learning modules that require flwr.
    This prevents import errors when flwr is not installed.
    """
    mock_utils = Mock()
    mock_utils.read_configs_to_toml = Mock(return_value={})

    mock_toml_adjustment = Mock()
    mock_toml_adjustment.update_flwr_config = Mock()

    modules_to_mock = {
        "federated_pneumonia_detection.src.control.federated_new_version.core.utils": mock_utils,
        "federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment": mock_toml_adjustment,
    }

    # Add mocks to sys.modules
    for module_name, mock_module in modules_to_mock.items():
        sys.modules[module_name] = mock_module

    yield

    # Clean up sys.modules
    for module_name in modules_to_mock.keys():
        sys.modules.pop(module_name, None)


class TestRunFederatedTrainingTask:
    """Test run_federated_training_task function."""

    @pytest.mark.unit
    def test_federated_training_task_success(
        self,
        mock_config_manager,
        mock_subprocess_popen,
        tmp_path,
    ):
        """Test successful federated training task."""
        # Create test data structure
        source_path = tmp_path / "data"
        images_dir = source_path / "Images"
        images_dir.mkdir(parents=True)
        csv_path = source_path / "metadata.csv"
        csv_path.write_text("file,label\nimg.jpg,0\n")

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.os.environ.copy",
                return_value={
                    "POSTGRES_DB_URI": "test",
                    "POSTGRES_DB": "test",
                    "POSTGRES_USER": "test",
                    "POSTGRES_PASSWORD": "test",
                },
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path.exists",
                return_value=True,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path",
                Path,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.subprocess.Popen",
                return_value=mock_subprocess_popen,
            ),
            patch(
                "federated_pneumonia_detection.src.control.federated_new_version.core.utils.read_configs_to_toml",
                return_value={"num_server_rounds": 3},
            ),
        ):
            result = run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                num_server_rounds=3,
            )

            assert result["status"] == "completed"
            assert result["experiment_name"] == "test_federated"
            assert result["return_code"] == 0

    @pytest.mark.unit
    def test_federated_training_task_csv_not_found(
        self,
        mock_config_manager,
        tmp_path,
    ):
        """Test error handling when CSV file is missing."""
        source_path = tmp_path / "data"
        images_dir = source_path / "Images"
        images_dir.mkdir(parents=True)
        # No CSV file

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
            return_value=mock_config_manager,
        ):
            result = run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                num_server_rounds=3,
            )

            assert result["status"] == "failed"
            assert "CSV file not found" in result["error"]

    @pytest.mark.unit
    def test_federated_training_task_images_not_found(
        self,
        mock_config_manager,
        tmp_path,
    ):
        """Test error handling when Images directory is missing."""
        source_path = tmp_path / "data"
        source_path.mkdir(parents=True, exist_ok=True)
        csv_path = source_path / "metadata.csv"
        csv_path.write_text("file,label\n")

        # No Images directory

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
            return_value=mock_config_manager,
        ):
            result = run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                num_server_rounds=3,
            )

            assert result["status"] == "failed"
            assert "Images directory not found" in result["error"]

    @pytest.mark.unit
    def test_federated_training_task_config_updates(
        self,
        mock_config_manager,
        mock_subprocess_popen,
        tmp_path,
    ):
        """Test that configuration is properly updated."""
        source_path = tmp_path / "data"
        images_dir = source_path / "Images"
        images_dir.mkdir(parents=True)
        csv_path = source_path / "metadata.csv"
        csv_path.write_text("file,label\n")

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.os.environ.copy",
                return_value={
                    "POSTGRES_DB_URI": "test",
                    "POSTGRES_DB": "test",
                    "POSTGRES_USER": "test",
                    "POSTGRES_PASSWORD": "test",
                },
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path.exists",
                return_value=True,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path",
                Path,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.subprocess.Popen",
                return_value=mock_subprocess_popen,
            ),
            patch(
                "federated_pneumonia_detection.src.control.federated_new_version.core.utils.read_configs_to_toml",
                return_value={},
            ),
        ):
            run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                num_server_rounds=5,
            )

            # Verify config.set was called
            mock_config_manager.set.assert_any_call(
                "experiment.file-path",
                str(csv_path),
            )
            mock_config_manager.set.assert_any_call(
                "experiment.image-dir",
                str(images_dir),
            )
            mock_config_manager.set.assert_any_call("experiment.num-server-rounds", 5)

    @pytest.mark.unit
    def test_federated_training_task_subprocess_failure(
        self,
        mock_config_manager,
        tmp_path,
    ):
        """Test handling when subprocess fails (non-zero return code)."""
        source_path = tmp_path / "data"
        images_dir = source_path / "Images"
        images_dir.mkdir(parents=True)
        csv_path = source_path / "metadata.csv"
        csv_path.write_text("file,label\n")

        mock_process = Mock()
        mock_process.pid = 99999
        mock_process.wait.return_value = 1  # Non-zero return code
        mock_process.stdout = None

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.os.environ.copy",
                return_value={
                    "POSTGRES_DB_URI": "test",
                    "POSTGRES_DB": "test",
                    "POSTGRES_USER": "test",
                    "POSTGRES_PASSWORD": "test",
                },
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path.exists",
                return_value=True,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path",
                Path,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.subprocess.Popen",
                return_value=mock_process,
            ),
            patch(
                "federated_pneumonia_detection.src.control.federated_new_version.core.utils.read_configs_to_toml",
                return_value={},
            ),
        ):
            result = run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                num_server_rounds=3,
            )

            assert result["status"] == "failed"
            assert result["return_code"] == 1

    @pytest.mark.unit
    def test_federated_training_task_rf_ps1_not_found(
        self,
        mock_config_manager,
        tmp_path,
    ):
        """Test error handling when rf.ps1 script is not found."""
        source_path = tmp_path / "data"
        images_dir = source_path / "Images"
        images_dir.mkdir(parents=True)
        csv_path = source_path / "metadata.csv"
        csv_path.write_text("file,label\n")

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path.exists",
                side_effect=lambda *args: any(
                    str(arg).endswith("Images") or str(arg).endswith("metadata.csv")
                    for arg in args
                ),
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path",
                Path,
            ),
        ):
            result = run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                num_server_rounds=3,
            )

            assert result["status"] == "failed"
            assert "rf.ps1 script not found" in result["error"]

    @pytest.mark.unit
    def test_federated_training_task_default_rounds(
        self,
        mock_config_manager,
        mock_subprocess_popen,
        tmp_path,
    ):
        """Test that default num_server_rounds (3) is used."""
        source_path = tmp_path / "data"
        images_dir = source_path / "Images"
        images_dir.mkdir(parents=True)
        csv_path = source_path / "metadata.csv"
        csv_path.write_text("file,label\n")

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.os.environ.copy",
                return_value={
                    "POSTGRES_DB_URI": "test",
                    "POSTGRES_DB": "test",
                    "POSTGRES_USER": "test",
                    "POSTGRES_PASSWORD": "test",
                },
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path.exists",
                return_value=True,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path",
                Path,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.subprocess.Popen",
                return_value=mock_subprocess_popen,
            ),
            patch(
                "federated_pneumonia_detection.src.control.federated_new_version.core.utils.read_configs_to_toml",
                return_value={},
            ),
        ):
            result = run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                # num_server_rounds not provided - should default to 3
            )

            mock_config_manager.set.assert_any_call("experiment.num-server-rounds", 3)

    @pytest.mark.unit
    def test_federated_training_task_toml_sync(
        self,
        mock_config_manager,
        mock_subprocess_popen,
        tmp_path,
    ):
        """Test that config is synced to pyproject.toml."""
        source_path = tmp_path / "data"
        images_dir = source_path / "Images"
        images_dir.mkdir(parents=True)
        csv_path = source_path / "metadata.csv"
        csv_path.write_text("file,label\n")

        mock_configs = {"num_server_rounds": 3, "max_epochs": 2}

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.os.environ.copy",
                return_value={
                    "POSTGRES_DB_URI": "test",
                    "POSTGRES_DB": "test",
                    "POSTGRES_USER": "test",
                    "POSTGRES_PASSWORD": "test",
                },
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path.exists",
                return_value=True,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path",
                Path,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.subprocess.Popen",
                return_value=mock_subprocess_popen,
            ),
            patch(
                "federated_pneumonia_detection.src.control.federated_new_version.core.utils.read_configs_to_toml",
                return_value=mock_configs,
            ) as mock_read_configs,
            patch(
                "federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment.update_flwr_config",
            ) as mock_update_flwr,
        ):
            run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                num_server_rounds=3,
            )

            mock_read_configs.assert_called_once()
            mock_update_flwr.assert_called_once_with(**mock_configs)

    @pytest.mark.unit
    def test_federated_training_task_no_toml_configs(
        self,
        mock_config_manager,
        mock_subprocess_popen,
        tmp_path,
    ):
        """Test behavior when no configs to sync to TOML."""
        source_path = tmp_path / "data"
        images_dir = source_path / "Images"
        images_dir.mkdir(parents=True)
        csv_path = source_path / "metadata.csv"
        csv_path.write_text("file,label\n")

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.os.environ.copy",
                return_value={
                    "POSTGRES_DB_URI": "test",
                    "POSTGRES_DB": "test",
                    "POSTGRES_USER": "test",
                    "POSTGRES_PASSWORD": "test",
                },
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path.exists",
                return_value=True,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path",
                Path,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.subprocess.Popen",
                return_value=mock_subprocess_popen,
            ),
            patch(
                "federated_pneumonia_detection.src.control.federated_new_version.core.utils.read_configs_to_toml",
                return_value=None,
            ),
            patch(
                "federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment.update_flwr_config",
            ) as mock_update_flwr,
        ):
            result = run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                num_server_rounds=3,
            )

            # Should not call update_flwr_config when no configs
            mock_update_flwr.assert_not_called()
            assert result["status"] == "completed"

    @pytest.mark.unit
    def test_federated_training_task_environment_variables(
        self,
        mock_config_manager,
        mock_subprocess_popen,
        tmp_path,
    ):
        """Test that environment variables are passed to subprocess."""
        source_path = tmp_path / "data"
        images_dir = source_path / "Images"
        images_dir.mkdir(parents=True)
        csv_path = source_path / "metadata.csv"
        csv_path.write_text("file,label\n")

        test_env = {
            "POSTGRES_DB_URI": "postgresql://localhost/test",
            "POSTGRES_DB": "test_db",
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_password",
            "CUSTOM_VAR": "custom_value",
        }

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.os.environ.copy",
                return_value=test_env,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path.exists",
                return_value=True,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path",
                Path,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.subprocess.Popen",
                return_value=mock_subprocess_popen,
            ) as mock_popen,
            patch(
                "federated_pneumonia_detection.src.control.federated_new_version.core.utils.read_configs_to_toml",
                return_value={},
            ),
        ):
            run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                num_server_rounds=3,
            )

            # Verify Popen was called with env parameter
            call_kwargs = mock_popen.call_args[1]
            assert "env" in call_kwargs
            assert call_kwargs["env"] == test_env

    @pytest.mark.unit
    def test_federated_training_task_missing_env_vars(
        self,
        mock_config_manager,
        mock_subprocess_popen,
        tmp_path,
        monkeypatch,
        caplog,
    ):
        """Test warning when required environment variables are missing."""
        import logging

        caplog.set_level(logging.WARNING)

        source_path = tmp_path / "data"
        images_dir = source_path / "Images"
        images_dir.mkdir(parents=True)
        csv_path = source_path / "metadata.csv"
        csv_path.write_text("file,label\n")

        # Unset all required env vars to simulate missing environment
        for var in ["POSTGRES_DB_URI", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]:
            monkeypatch.delenv(var, raising=False)

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path.exists",
                return_value=True,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path",
                Path,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.subprocess.Popen",
                return_value=mock_subprocess_popen,
            ),
        ):
            run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                num_server_rounds=3,
            )

            # Check for warning about missing env vars using caplog
            log_messages = [record.message for record in caplog.records]
            assert any("Missing environment variables" in msg for msg in log_messages)

    @pytest.mark.unit
    def test_federated_training_task_unexpected_exception(
        self,
        mock_config_manager,
        tmp_path,
    ):
        """Test handling of unexpected exceptions."""
        source_path = tmp_path / "data"
        images_dir = source_path / "Images"
        images_dir.mkdir(parents=True)
        csv_path = source_path / "metadata.csv"
        csv_path.write_text("file,label\n")

        with (
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.ConfigManager",
                return_value=mock_config_manager,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path.exists",
                return_value=True,
            ),
            patch(
                "federated_pneumonia_detection.src.api.endpoints.experiments.utils.federated_tasks.Path",
                Path,
            ),
            patch(
                "federated_pneumonia_detection.src.control.federated_new_version.core.utils.read_configs_to_toml",
                side_effect=RuntimeError("Unexpected error"),
            ),
        ):
            result = run_federated_training_task(
                source_path=str(source_path),
                experiment_name="test_federated",
                csv_filename="metadata.csv",
                num_server_rounds=3,
            )

            assert result["status"] == "failed"
            assert "Unexpected error" in result["error"]
