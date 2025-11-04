"""
Federated training task functions.

Provides background task execution for federated machine learning training
using Flower (flwr) framework via rf.ps1 script.
"""

import subprocess
import os
from typing import Any, Dict
from pathlib import Path

from federated_pneumonia_detection.src.utils.loggers.logger import get_logger
from federated_pneumonia_detection.config.config_manager import ConfigManager


def run_federated_training_task(
    source_path: str,
    experiment_name: str,
    csv_filename: str,
    num_server_rounds: int = 3,
) -> Dict[str, Any]:
    """
    Background task to execute federated training via rf.ps1 script.

    Updates configuration with the uploaded dataset paths and spawns the
    federated training process using the PowerShell script.

    Args:
        source_path: Path to extracted training data directory
        experiment_name: Name identifier for this training run
        csv_filename: Name of the CSV metadata file within the data directory
        num_server_rounds: Number of federated learning rounds (default: 3)

    Returns:
        Dictionary containing training status and tracking information
    """
    task_logger = get_logger(f"{__name__}._task")

    task_logger.info("=" * 80)
    task_logger.info("FEDERATED TRAINING - Pneumonia Detection (Background Task)")
    task_logger.info("=" * 80)

    try:
        task_logger.info(f"  Source: {source_path}")
        task_logger.info(f"  Experiment: {experiment_name}")
        task_logger.info(f"  CSV File: {csv_filename}")
        task_logger.info(f"  Server Rounds: {num_server_rounds}")

        # Construct the full paths for training data
        csv_path = os.path.join(source_path, csv_filename)
        image_dir = os.path.join(source_path, "Images")

        # Verify paths exist
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Images directory not found: {image_dir}")

        task_logger.info(f"  CSV Path: {csv_path}")
        task_logger.info(f"  Image Dir: {image_dir}")

        # Update configuration with uploaded data paths
        task_logger.info("\nUpdating configuration...")
        config_path = r"C:\Users\User\Projects\FYP2\federated_pneumonia_detection\config\default_config.yaml"
        config_manager = ConfigManager(config_path=config_path)

        # Store current config for logging
        old_file_path = config_manager.get("experiment.file-path", "N/A")
        old_image_dir = config_manager.get("experiment.image-dir", "N/A")
        old_rounds = config_manager.get("experiment.num-server-rounds", "N/A")

        # Update paths and rounds
        config_manager.set("experiment.file-path", csv_path)
        config_manager.set("experiment.image-dir", image_dir)
        config_manager.set("experiment.num-server-rounds", num_server_rounds)
        config_manager.save()

        task_logger.info(f"  Old file-path: {old_file_path}")
        task_logger.info(f"  New file-path: {csv_path}")
        task_logger.info(f"  Old image-dir: {old_image_dir}")
        task_logger.info(f"  New image-dir: {image_dir}")
        task_logger.info(f"  Old num-server-rounds: {old_rounds}")
        task_logger.info(f"  New num-server-rounds: {num_server_rounds}")

        # Prepare environment for rf.ps1 execution
        task_logger.info("\nPreparing federated training environment...")
        project_root = Path(__file__).parent.parent.parent.parent.parent.absolute()
        task_logger.info(f"  Project root: {project_root}")

        rf_script_path = project_root / "rf.ps1"
        if not rf_script_path.exists():
            raise FileNotFoundError(f"rf.ps1 script not found at: {rf_script_path}")

        task_logger.info(f"  Script: {rf_script_path}")

        # Execute rf.ps1 PowerShell script
        task_logger.info("\n" + "=" * 80)
        task_logger.info("STARTING FEDERATED TRAINING PROCESS")
        task_logger.info("=" * 80)

        ps_cmd = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(rf_script_path),
        ]

        task_logger.info(f"Executing: {' '.join(ps_cmd)}")
        task_logger.info(f"Working Directory: {project_root}")

        process = subprocess.Popen(
            ps_cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        task_logger.info(f"Process started with PID: {process.pid}")

        # Stream output while process runs
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                if line:
                    task_logger.info(f"[FLWR] {line.rstrip()}")

        # Wait for process to complete
        return_code = process.wait()

        task_logger.info("=" * 80)
        task_logger.info(
            f"Federated training process completed with return code: {return_code}"
        )
        task_logger.info("=" * 80)

        return {
            "message": "Federated training completed",
            "experiment_name": experiment_name,
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "source_path": source_path,
            "csv_filename": csv_filename,
        }

    except FileNotFoundError as e:
        task_logger.error(f"File not found error: {str(e)}")
        return {
            "message": f"Federated training failed: {str(e)}",
            "experiment_name": experiment_name,
            "status": "failed",
            "error": str(e),
        }
    except Exception as e:
        task_logger.error(
            f"Unexpected error during federated training: {str(e)}", exc_info=True
        )
        return {
            "message": f"Federated training failed: {str(e)}",
            "experiment_name": experiment_name,
            "status": "failed",
            "error": str(e),
        }
