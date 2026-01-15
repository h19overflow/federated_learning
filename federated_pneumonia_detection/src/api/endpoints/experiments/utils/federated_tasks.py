"""Federated training task functions.

Provides background task execution for federated machine learning training
using Flower (flwr) framework via `rf.ps1` script.

This module is invoked from FastAPI background tasks and should remain thin:
- validate paths
- update config + sync TOML
- spawn the PowerShell process
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.federated_new_version.core.utils import (
    read_configs_to_toml,
)
from federated_pneumonia_detection.src.control.federated_new_version.toml_adjustment import (
    update_flwr_config,
)


task_logger = logging.getLogger(__name__)


_REQUIRED_DB_ENV_VARS: tuple[str, ...] = (
    "POSTGRES_DB_URI",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
)


def _resolve_training_paths(source_path: str, csv_filename: str) -> tuple[str, str]:
    csv_path = os.path.join(source_path, csv_filename)
    image_dir = os.path.join(source_path, "Images")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Images directory not found: {image_dir}")

    return csv_path, image_dir


def _resolve_default_config_path() -> Path:
    return (
        Path(__file__).parent.parent.parent.parent.parent.parent
        / "config"
        / "default_config.yaml"
    )


def _update_experiment_config(
    *,
    csv_path: str,
    image_dir: str,
    num_server_rounds: int,
) -> None:
    config_path = _resolve_default_config_path()
    config_manager = ConfigManager(config_path=str(config_path))

    config_manager.set("experiment.file-path", csv_path)
    config_manager.set("experiment.image-dir", image_dir)
    config_manager.set("experiment.num-server-rounds", num_server_rounds)
    config_manager.save()


def _sync_flower_toml_configs() -> None:
    flwr_configs = read_configs_to_toml()
    if not flwr_configs:
        return

    update_flwr_config(**flwr_configs)


def _resolve_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.parent.absolute()


def _resolve_rf_script_path(project_root: Path) -> Path:
    rf_script_path = project_root / "rf.ps1"
    if not rf_script_path.exists():
        raise FileNotFoundError(f"rf.ps1 script not found at: {rf_script_path}")
    return rf_script_path


def _collect_subprocess_env() -> dict[str, str]:
    # Copy current environment (includes .env vars if FastAPI loaded them)
    return os.environ.copy()


def _warn_on_missing_db_env_vars(env: dict[str, str]) -> None:
    missing_vars = [var for var in _REQUIRED_DB_ENV_VARS if var not in env]
    if missing_vars:
        task_logger.warning(
            "Missing environment variables: %s. Database persistence may fail!",
            missing_vars,
        )


def _stream_subprocess_stdout(process: subprocess.Popen[str]) -> None:
    if not process.stdout:
        return

    for line in iter(process.stdout.readline, ""):
        if line:
            task_logger.info("[FLWR] %s", line.rstrip())


def _run_rf_ps1(project_root: Path, rf_script_path: Path, env: dict[str, str]) -> int:
    ps_cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(rf_script_path),
    ]

    process = subprocess.Popen(
        ps_cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    task_logger.info("Federated process started (pid=%s)", process.pid)
    _stream_subprocess_stdout(process)

    return process.wait()


def run_federated_training_task(
    source_path: str,
    experiment_name: str,
    csv_filename: str,
    num_server_rounds: int = 3,
) -> dict[str, Any]:
    """Background task to execute federated training via `rf.ps1`.

    Args:
        source_path: Path to extracted training data directory.
        experiment_name: Name identifier for this training run.
        csv_filename: Name of the CSV metadata file within the data directory.
        num_server_rounds: Number of federated learning rounds.

    Returns:
        Training status and tracking information.
    """

    try:
        csv_path, image_dir = _resolve_training_paths(source_path, csv_filename)

        _update_experiment_config(
            csv_path=csv_path,
            image_dir=image_dir,
            num_server_rounds=num_server_rounds,
        )
        _sync_flower_toml_configs()

        project_root = _resolve_project_root()
        rf_script_path = _resolve_rf_script_path(project_root)

        env = _collect_subprocess_env()
        _warn_on_missing_db_env_vars(env)

        return_code = _run_rf_ps1(project_root, rf_script_path, env)

        return {
            "message": "Federated training completed",
            "experiment_name": experiment_name,
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "source_path": source_path,
            "csv_filename": csv_filename,
        }

    except FileNotFoundError as exc:
        task_logger.error("Federated training failed (missing file): %s", exc)
        return {
            "message": f"Federated training failed: {exc}",
            "experiment_name": experiment_name,
            "status": "failed",
            "error": str(exc),
        }
    except Exception as exc:
        task_logger.error("Unexpected error during federated training", exc_info=True)
        return {
            "message": f"Federated training failed: {exc}",
            "experiment_name": experiment_name,
            "status": "failed",
            "error": str(exc),
        }
