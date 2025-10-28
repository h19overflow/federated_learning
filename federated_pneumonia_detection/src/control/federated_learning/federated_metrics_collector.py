"""
Federated learning metrics collector for tracking client-side training metrics.

This module provides comprehensive metrics collection for federated learning clients,
tracking metrics across federated rounds, local epochs, and aggregating results.

Dependencies:
- json, pandas: For metrics persistence
- datetime, pathlib: For file management

Role in System:
- Tracks per-round metrics for each federated learning client
- Records local training and evaluation metrics
- Aggregates client-level performance across rounds
- Persists metrics to JSON/CSV for analysis
- Streams metrics to frontend via WebSocket for real-time monitoring
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.control.dl_model.utils.data.websocket_metrics_sender import MetricsWebSocketSender
# TODO : client_complete is never sent to the frontend client , need to debug and figure out why ,
#  other events such as client_progress and round_start are working fine, it's not even logged on the python logs.
class FederatedMetricsCollector:
    """
    Comprehensive metrics collector for federated learning clients.

    Tracks metrics across federated rounds and local training epochs,
    saves to JSON and CSV for analysis and visualization.
    """

    def __init__(
        self,
        save_dir: str,
        client_id: str,
        experiment_name: str = "federated_experiment",
        run_id: Optional[int] = None,
        enable_db_persistence: bool = True,
        enable_progress_logging: bool = True,
        websocket_uri: Optional[str] = "ws://localhost:8765",
    ):
        """
        Initialize federated metrics collector.

        Args:
            save_dir: Directory to save metrics files
            client_id: Unique identifier for this client
            experiment_name: Name of the federated experiment
            run_id: Optional database run ID for persistence
            enable_db_persistence: Whether to save metrics to database
            enable_progress_logging: Whether to enable real-time progress logging
            websocket_uri: Optional WebSocket URI for real-time metrics streaming
        """
        self.save_dir = Path(save_dir)
        self.client_id = client_id
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.enable_db_persistence = enable_db_persistence
        self.enable_progress_logging = enable_progress_logging
        self.logger = logging.getLogger(__name__)

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize WebSocket sender for real-time metrics streaming
        self.ws_sender = None
        if websocket_uri:
            try:
                self.ws_sender = MetricsWebSocketSender(websocket_uri)
                self.logger.info(f"[FederatedMetrics] WebSocket sender initialized for client {client_id}")
            except Exception as e:
                self.logger.warning(f"[FederatedMetrics] Failed to initialize WebSocket sender: {e}")
                self.ws_sender = None

        # Metrics storage
        self.round_metrics = []  # One entry per federated round
        self.local_epoch_metrics = []  # Detailed local training metrics
        self.training_start_time = None
        self.training_end_time = None

        # Metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'client_id': client_id,
            'start_time': None,
            'end_time': None,
            'total_rounds': 0,
            'total_local_epochs': 0,
            'best_round': None,
            'best_val_accuracy': 0.0,
            'best_val_loss': float('inf'),
            'total_samples_trained': 0
        }

        self.logger.info(
            f"Federated metrics collector initialized for client {client_id}"
        )

    def start_training(self, model_info: Optional[Dict[str, Any]] = None):
        """
        Record training start time and model information.

        Args:
            model_info: Optional dictionary with model metadata
        """
        self.training_start_time = datetime.now()
        self.metadata['start_time'] = self.training_start_time.isoformat()

        if model_info:
            self.metadata.update(model_info)

        self.logger.info(f"Training started for client {self.client_id}")

    def record_round_start(self, round_num: int, server_config: Dict[str, Any]):
        """
        Record the start of a federated round.

        Args:
            round_num: Current federated round number
            server_config: Configuration received from server
        """
        round_data = {
            'round': round_num,
            'start_time': datetime.now().isoformat(),
            'server_config': server_config,
            'local_epochs': [],
            'fit_metrics': {},
            'eval_metrics': {}
        }

        self.round_metrics.append(round_data)

        # Send round start notification via WebSocket
        if self.ws_sender:
            try:
                total_rounds = server_config.get('num_rounds', round_num + 1)
                local_epochs = server_config.get('local_epochs', 1)
                self.ws_sender.send_metrics({
                    "run_id": self.run_id,
                    "round": round_num + 1,  # 1-indexed for display
                    "round_index": round_num,  # Keep 0-indexed for internal tracking
                    "total_rounds": total_rounds,
                    "client_id": self.client_id,
                    "experiment_name": self.experiment_name,
                    "local_epochs": local_epochs,
                    "status": "starting_round",
                    "timestamp": datetime.now().isoformat()
                }, "round_start")
                self.logger.info(f"[Client {self.client_id}] Sent round_start event for round {round_num + 1}/{total_rounds} with {local_epochs} local epochs")
            except Exception as e:
                self.logger.warning(f"Failed to send round_start via WebSocket: {e}")

        self.logger.info(f"Round {round_num} started for client {self.client_id}, local_epochs={server_config.get('local_epochs', 1)}")

    def record_local_epoch(
        self,
        round_num: int,
        local_epoch: int,
        train_loss: float,
        learning_rate: float,
        num_samples: int,
        additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Record metrics for a single local training epoch.

        Args:
            round_num: Current federated round
            local_epoch: Local epoch number within the round
            train_loss: Training loss for this epoch
            learning_rate: Learning rate used
            num_samples: Number of samples processed
            additional_metrics: Optional additional metrics
        """
        epoch_data = {
            'round': round_num,
            'local_epoch': local_epoch,
            'train_loss': train_loss,
            'learning_rate': learning_rate,
            'num_samples': num_samples,
            'timestamp': datetime.now().isoformat()
        }

        if additional_metrics:
            epoch_data.update(additional_metrics)

        self.local_epoch_metrics.append(epoch_data)

        # Update current round's local epochs
        if self.round_metrics and self.round_metrics[-1]['round'] == round_num:
            self.round_metrics[-1]['local_epochs'].append(epoch_data)

        # Send local epoch progress via WebSocket (optional - can be disabled for performance)
        if self.ws_sender and self.enable_progress_logging:
            try:
                metrics = {
                    'train_loss': train_loss,
                    'learning_rate': learning_rate,
                    'num_samples': num_samples
                }
                if additional_metrics:
                    metrics.update(additional_metrics)

                self.ws_sender.send_metrics({
                    "run_id": self.run_id,
                    "round": round_num + 1,  # 1-indexed for display
                    "round_index": round_num,  # Keep 0-indexed for internal tracking
                    "client_id": self.client_id,
                    "local_epoch": local_epoch,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                    "status": "training_progress",
                    "experiment_name": self.experiment_name
                }, "client_progress")
                self.logger.info(
                    f"[Client {self.client_id}] Round {round_num + 1}, Local Epoch {local_epoch}: "
                    f"Loss={train_loss:.4f}, Samples={num_samples}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to send client_progress via WebSocket: {e}")

    def record_fit_metrics(
        self,
        round_num: int,
        train_loss: float,
        num_samples: int,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Record metrics after local training (fit) completes.

        Args:
            round_num: Current federated round
            train_loss: Final training loss
            num_samples: Number of training samples
            additional_metrics: Optional additional metrics
        """
        fit_data = {
            'train_loss': train_loss,
            'num_samples': num_samples,
            'timestamp': datetime.now().isoformat()
        }

        if additional_metrics:
            fit_data.update(additional_metrics)

        # Update current round
        if self.round_metrics and self.round_metrics[-1]['round'] == round_num:
            self.round_metrics[-1]['fit_metrics'] = fit_data
            self.metadata['total_samples_trained'] += num_samples

        self.logger.info(
            f"Round {round_num}: Fit completed - Loss: {train_loss:.4f}, "
            f"Samples: {num_samples}"
        )

    def record_eval_metrics(
        self,
        round_num: int,
        val_loss: float,
        val_accuracy: float,
        num_samples: int,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Record evaluation metrics after model evaluation.

        Args:
            round_num: Current federated round
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            num_samples: Number of validation samples
            additional_metrics: Optional additional metrics (precision, recall, etc.)
        """
        eval_data = {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'num_samples': num_samples,
            'timestamp': datetime.now().isoformat()
        }

        if additional_metrics:
            eval_data.update(additional_metrics)

        # Update current round
        if self.round_metrics and self.round_metrics[-1]['round'] == round_num:
            self.round_metrics[-1]['eval_metrics'] = eval_data
            self.round_metrics[-1]['end_time'] = datetime.now().isoformat()

            # Send round end notification via WebSocket with both fit and eval metrics
            if self.ws_sender:
                try:
                    fit_metrics = self.round_metrics[-1].get('fit_metrics', {})
                    self.ws_sender.send_round_end(
                        round_num=round_num,
                        total_rounds=self.metadata.get('total_rounds', round_num + 1),
                        fit_metrics=fit_metrics,
                        eval_metrics=eval_data
                    )
                    self.logger.debug(
                        f"[Client {self.client_id}] Sent round_end event for round {round_num}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to send round_end via WebSocket: {e}")

        # Update best metrics
        if val_accuracy > self.metadata['best_val_accuracy']:
            self.metadata['best_val_accuracy'] = val_accuracy
            self.metadata['best_round'] = round_num

        if val_loss < self.metadata['best_val_loss']:
            self.metadata['best_val_loss'] = val_loss

        self.logger.info(
            f"Round {round_num}: Eval completed - Loss: {val_loss:.4f}, "
            f"Accuracy: {val_accuracy:.4f}"
        )
  # TODO: Lastly we need to format this well so it follos the run_metrics schema in the engine , we need to ensure the persestince is valid

    def end_training(self):
        """Record training end time and save all metrics."""
        self.training_end_time = datetime.now()
        self.metadata['end_time'] = self.training_end_time.isoformat()
        self.metadata['total_rounds'] = len(self.round_metrics)
        self.metadata['total_local_epochs'] = len(self.local_epoch_metrics)

        if self.training_start_time:
            duration = self.training_end_time - self.training_start_time
            self.metadata['training_duration_seconds'] = duration.total_seconds()
            self.metadata['training_duration_formatted'] = str(duration)

        # Send client training completion via WebSocket
        if self.ws_sender:
            try:
                self.ws_sender.send_metrics({
                    "run_id": self.run_id,
                    "client_id": self.client_id,
                    "status": "completed",
                    "total_rounds": self.metadata['total_rounds'],
                    "total_local_epochs": self.metadata['total_local_epochs'],
                    "best_round": self.metadata['best_round'],
                    "best_val_accuracy": self.metadata['best_val_accuracy'],
                    "best_val_loss": self.metadata['best_val_loss'],
                    "total_samples_trained": self.metadata['total_samples_trained'],
                    "training_duration": self.metadata.get('training_duration_formatted', '0:00:00')
                }, "client_complete")
                self.logger.info(
                    f"[Client {self.client_id}] Sent client_complete event"
                )
            except Exception as e:
                self.logger.warning(f"Failed to send client_complete via WebSocket: {e}")


    def get_round_metrics(self) -> List[Dict[str, Any]]:
        """Return round metrics history."""
        return self.round_metrics

    def get_local_epoch_metrics(self) -> List[Dict[str, Any]]:
        """Return local epoch metrics history."""
        return self.local_epoch_metrics

    def get_metadata(self) -> Dict[str, Any]:
        """Return experiment metadata."""
        return self.metadata
