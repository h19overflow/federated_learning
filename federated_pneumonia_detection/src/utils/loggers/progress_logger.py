"""
Multiprocessing-safe progress logger for training observability.

Provides file-based logging of training progress that's safe for multiprocessing
and can be easily adapted to WebSocket broadcasting for frontend integration.

Dependencies:
- json, logging: For structured logging
- pathlib, datetime: For file management
- threading.Lock: For thread-safe file writes

Role in System:
- Logs training progress to JSON file for frontend consumption
- Multiprocessing-safe (no async/pickle issues)
- Provides epoch-level metrics in real-time
- Can be replaced with WebSocket logger when frontend is ready
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from threading import Lock


class ProgressLogger:
    """
    File-based progress logger for training metrics.

    Logs training progress to a JSON file that can be monitored by the frontend.
    Thread-safe and multiprocessing-safe (no pickling issues).
    """

    def __init__(
        self,
        log_file: str = "training_progress.json",
        log_dir: str = "logs/progress",
        experiment_name: str = "experiment",
        mode: str = "centralized"
    ):
        """
        Initialize progress logger.

        Args:
            log_file: Name of the progress log file
            log_dir: Directory to store progress logs
            experiment_name: Name of the experiment
            mode: Training mode ('centralized' or 'federated')
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{mode}_{timestamp}.json"

        # Thread-safe file writing
        self._lock = Lock()
        self.logger = logging.getLogger(__name__)

        # Initialize log file with metadata
        self._init_log_file(experiment_name, mode)

        self.logger.info(f"Progress logger initialized: {self.log_file}")

    def _init_log_file(self, experiment_name: str, mode: str):
        """Initialize the log file with metadata."""
        initial_data = {
            "metadata": {
                "experiment_name": experiment_name,
                "training_mode": mode,
                "start_time": datetime.now().isoformat(),
                "status": "started"
            },
            "epochs": [],
            "current_epoch": None
        }

        with self._lock:
            with open(self.log_file, 'w') as f:
                json.dump(initial_data, f, indent=2)

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """
        Log the start of a training epoch.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
        """
        message = {
            "type": "epoch_start",
            "epoch": epoch,
            "total_epochs": total_epochs,
            "timestamp": datetime.now().isoformat()
        }

        self._append_to_log(message)
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")

    def log_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        phase: str = "train"
    ):
        """
        Log metrics at the end of an epoch.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics (loss, accuracy, etc.)
            phase: Training phase ('train', 'val', 'test')
        """
        message = {
            "type": "epoch_end",
            "epoch": epoch,
            "phase": phase,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

        self._append_to_log(message)

        # Log human-readable metrics
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        self.logger.info(f"Epoch {epoch} [{phase}] - {metrics_str}")

    def log_training_complete(self, final_metrics: Dict[str, Any]):
        """
        Log training completion.

        Args:
            final_metrics: Final training metrics and summary
        """
        message = {
            "type": "training_complete",
            "final_metrics": final_metrics,
            "timestamp": datetime.now().isoformat()
        }

        self._append_to_log(message)
        self._update_status("completed")
        self.logger.info("Training completed")

    def log_training_error(self, error_message: str):
        """
        Log training error.

        Args:
            error_message: Error message to log
        """
        message = {
            "type": "training_error",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }

        self._append_to_log(message)
        self._update_status("failed")
        self.logger.error(f"Training error: {error_message}")

    def log_custom_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log a custom event.

        Args:
            event_type: Type of event
            data: Event data
        """
        message = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        self._append_to_log(message)

    def _append_to_log(self, message: Dict[str, Any]):
        """
        Append a message to the log file (thread-safe).

        Args:
            message: Message to append
        """
        with self._lock:
            try:
                # Read current log
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)

                # Append new message to epochs list
                log_data["epochs"].append(message)

                # Update current epoch if applicable
                if "epoch" in message:
                    log_data["current_epoch"] = message["epoch"]

                # Write back
                with open(self.log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)

            except Exception as e:
                self.logger.error(f"Failed to append to log: {e}")

    def _update_status(self, status: str):
        """
        Update the training status in the log file.

        Args:
            status: New status ('started', 'running', 'completed', 'failed')
        """
        with self._lock:
            try:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)

                log_data["metadata"]["status"] = status
                log_data["metadata"]["end_time"] = datetime.now().isoformat()

                with open(self.log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)

            except Exception as e:
                self.logger.error(f"Failed to update status: {e}")

    def get_log_file_path(self) -> Path:
        """Get the path to the log file."""
        return self.log_file


class FederatedProgressLogger(ProgressLogger):
    """
    Extended progress logger for federated learning.

    Adds federated-specific logging for rounds and client metrics.
    """

    def __init__(
        self,
        client_id: str,
        log_file: Optional[str] = None,
        log_dir: str = "logs/progress",
        experiment_name: str = "federated_experiment"
    ):
        """
        Initialize federated progress logger.

        Args:
            client_id: Unique client identifier
            log_file: Optional custom log file name
            log_dir: Directory to store progress logs
            experiment_name: Name of the experiment
        """
        self.client_id = client_id

        if log_file is None:
            log_file = f"client_{client_id}_progress.json"

        super().__init__(
            log_file=log_file,
            log_dir=log_dir,
            experiment_name=experiment_name,
            mode="federated"
        )

    def log_round_start(self, round_num: int, total_rounds: int):
        """
        Log the start of a federated round.

        Args:
            round_num: Current round number
            total_rounds: Total number of rounds
        """
        message = {
            "type": "round_start",
            "round": round_num,
            "total_rounds": total_rounds,
            "client_id": self.client_id,
            "timestamp": datetime.now().isoformat()
        }

        self._append_to_log(message)
        self.logger.info(f"Client {self.client_id} - Round {round_num}/{total_rounds} started")

    def log_round_end(
        self,
        round_num: int,
        fit_metrics: Dict[str, float],
        eval_metrics: Dict[str, float]
    ):
        """
        Log metrics at the end of a federated round.

        Args:
            round_num: Current round number
            fit_metrics: Training (fit) metrics
            eval_metrics: Evaluation metrics
        """
        message = {
            "type": "round_end",
            "round": round_num,
            "client_id": self.client_id,
            "fit_metrics": fit_metrics,
            "eval_metrics": eval_metrics,
            "timestamp": datetime.now().isoformat()
        }

        self._append_to_log(message)

        # Log human-readable summary
        fit_str = ", ".join([f"{k}: {v:.4f}" for k, v in fit_metrics.items() if isinstance(v, (int, float))])
        eval_str = ", ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items() if isinstance(v, (int, float))])

        self.logger.info(
            f"Client {self.client_id} - Round {round_num} complete\n"
            f"  Fit: {fit_str}\n"
            f"  Eval: {eval_str}"
        )

    def log_local_epoch(
        self,
        round_num: int,
        local_epoch: int,
        metrics: Dict[str, float]
    ):
        """
        Log local epoch metrics during federated training.

        Args:
            round_num: Current federated round
            local_epoch: Local epoch number
            metrics: Epoch metrics
        """
        message = {
            "type": "local_epoch",
            "round": round_num,
            "local_epoch": local_epoch,
            "client_id": self.client_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

        self._append_to_log(message)
