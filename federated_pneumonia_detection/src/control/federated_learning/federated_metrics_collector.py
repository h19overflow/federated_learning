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
from federated_pneumonia_detection.src.utils.loggers.progress_logger import FederatedProgressLogger
from federated_pneumonia_detection.src.utils.loggers.websocket_progress_logger import (
    WebSocketFederatedProgressLogger,
)

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
        progress_log_dir: str = "logs/progress",
        websocket_manager: Optional[Any] = None,
        enable_websocket_broadcasting: bool = False,
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
            progress_log_dir: Directory for progress logs
            websocket_manager: Optional ConnectionManager for WebSocket broadcasting
            enable_websocket_broadcasting: Whether to broadcast progress via WebSocket
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

        # Initialize progress logger (multiprocessing-safe)
        self.progress_logger = None
        if enable_progress_logging:
            try:
                # Use WebSocket logger if manager and broadcasting are enabled
                if websocket_manager and enable_websocket_broadcasting:
                    self.progress_logger = WebSocketFederatedProgressLogger(
                        client_id=client_id,
                        websocket_manager=websocket_manager,
                        experiment_id=experiment_name,
                        log_dir=progress_log_dir,
                        experiment_name=experiment_name,
                        enable_file_logging=True,
                        broadcast_interval=0.5
                    )
                    self.logger.info(
                        f"WebSocket progress logging enabled for client {client_id}: "
                        f"{self.progress_logger.get_log_file_path()}"
                    )
                else:
                    self.progress_logger = FederatedProgressLogger(
                        client_id=client_id,
                        log_dir=progress_log_dir,
                        experiment_name=experiment_name
                    )
                    self.logger.info(
                        f"Progress logging enabled for client {client_id}: "
                        f"{self.progress_logger.get_log_file_path()}"
                    )
            except Exception as e:
                self.logger.warning(f"Could not initialize progress logger: {e}")
                self.progress_logger = None

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

        # Log progress for frontend observability
        if self.progress_logger:
            try:
                total_rounds = server_config.get('total_rounds', round_num)
                self.progress_logger.log_round_start(round_num, total_rounds)
            except Exception as e:
                self.logger.warning(f"Progress logging failed: {e}")

        self.logger.info(f"Round {round_num} started for client {self.client_id}")

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

        # Log progress for frontend observability
        if self.progress_logger:
            try:
                metrics = {
                    'train_loss': train_loss,
                    'learning_rate': learning_rate,
                    'num_samples': num_samples
                }
                if additional_metrics:
                    metrics.update(additional_metrics)

                self.progress_logger.log_local_epoch(
                    round_num=round_num,
                    local_epoch=local_epoch,
                    metrics=metrics
                )
            except Exception as e:
                self.logger.warning(f"Progress logging failed: {e}")

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

            # Log progress for frontend observability
            if self.progress_logger:
                try:
                    fit_metrics = self.round_metrics[-1].get('fit_metrics', {})
                    self.progress_logger.log_round_end(
                        round_num=round_num,
                        fit_metrics=fit_metrics,
                        eval_metrics=eval_data
                    )
                except Exception as e:
                    self.logger.warning(f"Progress logging failed: {e}")

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

        # Log training completion
        if self.progress_logger:
            try:
                self.progress_logger.log_training_complete({
                    'total_rounds': self.metadata['total_rounds'],
                    'total_local_epochs': self.metadata['total_local_epochs'],
                    'best_round': self.metadata['best_round'],
                    'best_val_accuracy': self.metadata['best_val_accuracy'],
                    'best_val_loss': self.metadata['best_val_loss'],
                    'total_samples_trained': self.metadata['total_samples_trained'],
                    'duration_seconds': self.metadata.get('training_duration_seconds', 0)
                })
            except Exception as e:
                self.logger.warning(f"Progress logging failed: {e}")

        # Save metrics
        self._save_metrics()
        self.logger.info(
            f"Training ended for client {self.client_id}. "
            f"Metrics saved to {self.save_dir}"
        )

    def _save_metrics(self):
        """Save all collected metrics to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        client_prefix = f"{self.experiment_name}_client_{self.client_id}"

        # Save comprehensive JSON
        json_path = self.save_dir / f"{client_prefix}_metrics_{timestamp}.json"
        full_data = {
            'metadata': self.metadata,
            'round_metrics': self.round_metrics,
            'local_epoch_metrics': self.local_epoch_metrics
        }

        with open(json_path, 'w') as f:
            json.dump(full_data, f, indent=2)
        self.logger.info(f"Saved JSON metrics to: {json_path}")

        # Save round metrics as CSV
        if self.round_metrics:
            round_df = self._flatten_round_metrics()
            csv_path = self.save_dir / f"{client_prefix}_rounds_{timestamp}.csv"
            round_df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved round metrics CSV to: {csv_path}")

        # Save local epoch metrics as CSV
        if self.local_epoch_metrics:
            epoch_df = pd.DataFrame(self.local_epoch_metrics)
            csv_path = self.save_dir / f"{client_prefix}_epochs_{timestamp}.csv"
            epoch_df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved epoch metrics CSV to: {csv_path}")

        # Save metadata
        metadata_path = self.save_dir / f"{client_prefix}_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        self.logger.info(f"Saved metadata to: {metadata_path}")

        # Create summary report
        self._create_summary_report(timestamp, client_prefix)

        # Persist to database if enabled
        if self.enable_db_persistence and self.run_id:
            try:
                self.persist_to_database()
            except Exception as e:
                self.logger.error(f"Database persistence failed: {e}")

    def _flatten_round_metrics(self) -> pd.DataFrame:
        """Flatten round metrics for CSV export."""
        flattened = []

        for round_data in self.round_metrics:
            flat_entry = {
                'round': round_data['round'],
                'start_time': round_data['start_time'],
                'end_time': round_data.get('end_time', None)
            }

            # Add fit metrics
            fit_metrics = round_data.get('fit_metrics', {})
            for key, value in fit_metrics.items():
                if key != 'timestamp':
                    flat_entry[f'fit_{key}'] = value

            # Add eval metrics
            eval_metrics = round_data.get('eval_metrics', {})
            for key, value in eval_metrics.items():
                if key != 'timestamp':
                    flat_entry[f'eval_{key}'] = value

            # Add local epoch summary
            local_epochs = round_data.get('local_epochs', [])
            if local_epochs:
                flat_entry['num_local_epochs'] = len(local_epochs)
                flat_entry['avg_local_loss'] = sum(
                    e['train_loss'] for e in local_epochs
                ) / len(local_epochs)

            flattened.append(flat_entry)

        return pd.DataFrame(flattened)

    def _create_summary_report(self, timestamp: str, client_prefix: str):
        """Create a human-readable summary report."""
        report_path = self.save_dir / f"{client_prefix}_summary_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"FEDERATED LEARNING SUMMARY - CLIENT {self.client_id}\n")
            f.write("=" * 80 + "\n\n")

            f.write("EXPERIMENT METADATA:\n")
            f.write("-" * 80 + "\n")
            for key, value in self.metadata.items():
                f.write(f"{key:30s}: {value}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("ROUND-BY-ROUND METRICS:\n")
            f.write("=" * 80 + "\n\n")

            for round_data in self.round_metrics:
                f.write(f"\nRound {round_data['round']}:\n")
                f.write("-" * 80 + "\n")

                # Fit metrics
                fit_metrics = round_data.get('fit_metrics', {})
                if fit_metrics:
                    f.write("  Training (Fit):\n")
                    for key, value in fit_metrics.items():
                        if key != 'timestamp' and isinstance(value, (int, float)):
                            f.write(f"    {key:26s}: {value:.6f}\n")

                # Eval metrics
                eval_metrics = round_data.get('eval_metrics', {})
                if eval_metrics:
                    f.write("  Evaluation:\n")
                    for key, value in eval_metrics.items():
                        if key != 'timestamp' and isinstance(value, (int, float)):
                            f.write(f"    {key:26s}: {value:.6f}\n")

                # Local epochs summary
                local_epochs = round_data.get('local_epochs', [])
                if local_epochs:
                    f.write(f"  Local Epochs: {len(local_epochs)}\n")

            # Best metrics summary
            f.write("\n" + "=" * 80 + "\n")
            f.write("BEST METRICS ACHIEVED:\n")
            f.write("=" * 80 + "\n")
            f.write(
                f"Best Validation Accuracy: {self.metadata['best_val_accuracy']:.6f} "
                f"(Round {self.metadata['best_round']})\n"
            )
            f.write(f"Best Validation Loss:     {self.metadata['best_val_loss']:.6f}\n")
            f.write(f"Total Samples Trained:    {self.metadata['total_samples_trained']}\n")

        self.logger.info(f"Saved summary report to: {report_path}")

    def get_round_metrics(self) -> List[Dict[str, Any]]:
        """Return round metrics history."""
        return self.round_metrics

    def get_local_epoch_metrics(self) -> List[Dict[str, Any]]:
        """Return local epoch metrics history."""
        return self.local_epoch_metrics

    def get_metadata(self) -> Dict[str, Any]:
        """Return experiment metadata."""
        return self.metadata

    def persist_to_database(self, db: Optional[Session] = None):
        """
        Persist collected federated metrics to database.

        Args:
            db: Optional database session. If None, creates a new session.
        """
        if not self.enable_db_persistence:
            self.logger.info("Database persistence is disabled")
            return

        if self.run_id is None:
            self.run_id = run_crud.create(db, **{
                'training_mode': 'federated',
                'status': 'in_progress',
                'start_time': datetime.now(),
                'wandb_id': 'placeholder',
                'source_path': 'placeholder',
            })
        
        close_session = False
        if db is None:
            db = get_session()
            close_session = True

        try:
            metrics_to_persist = []

            # Persist round-level metrics
            for round_data in self.round_metrics:
                round_num = round_data['round']

                # Persist fit metrics (training)
                fit_metrics = round_data.get('fit_metrics', {})
                for key, value in fit_metrics.items():
                    if key in ['timestamp'] or not isinstance(value, (int, float)):
                        continue

                    metrics_to_persist.append({
                        'run_id': self.run_id,
                        'metric_name': f'fit_{key}',
                        'metric_value': float(value),
                        'step': round_num,
                        'dataset_type': 'train'
                    })

                # Persist eval metrics (validation)
                eval_metrics = round_data.get('eval_metrics', {})
                for key, value in eval_metrics.items():
                    if key in ['timestamp'] or not isinstance(value, (int, float)):
                        continue

                    metrics_to_persist.append({
                        'run_id': self.run_id,
                        'metric_name': f'eval_{key}',
                        'metric_value': float(value),
                        'step': round_num,
                        'dataset_type': 'validation'
                    })

            # Persist local epoch metrics (more granular)
            for epoch_data in self.local_epoch_metrics:
                round_num = epoch_data.get('round', 0)
                local_epoch = epoch_data.get('local_epoch', 0)

                # Use combined step: round * 1000 + local_epoch for uniqueness
                combined_step = round_num * 1000 + local_epoch

                for key, value in epoch_data.items():
                    if key in ['round', 'local_epoch', 'timestamp']:
                        continue

                    if not isinstance(value, (int, float)):
                        continue

                    metrics_to_persist.append({
                        'run_id': self.run_id,
                        'metric_name': f'local_{key}',
                        'metric_value': float(value),
                        'step': combined_step,
                        'dataset_type': 'train'
                    })

            # Bulk create metrics for efficiency
            if metrics_to_persist:
                run_metric_crud.bulk_create(db, metrics_to_persist)
                db.commit()
                self.logger.info(
                    f"Persisted {len(metrics_to_persist)} federated metrics to database "
                    f"for client {self.client_id}, run_id={self.run_id}"
                )

        except Exception as e:
            self.logger.error(f"Failed to persist federated metrics to database: {e}")
            if close_session:
                db.rollback()
            raise
        finally:
            if close_session:
                db.close()
