"""
Training callbacks and utilities for PyTorch Lightning model training.
Provides checkpoint management, early stopping, and monitoring functionality.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig


class HighestValRecallCallback(pl.Callback):
    """Custom callback to track highest validation recall achieved during training."""

    def __init__(self):
        super().__init__()
        self.best_recall = 0.0
        self.logger = logging.getLogger(__name__)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Track the highest validation recall."""
        current_recall = trainer.callback_metrics.get('val_recall', 0.0)
        if isinstance(current_recall, torch.Tensor):
            current_recall = current_recall.item()

        if current_recall > self.best_recall:
            self.best_recall = current_recall
            self.logger.info(f"New best validation recall: {self.best_recall:.4f}")


class MetricsCollectorCallback(pl.Callback):
    """
    Comprehensive metrics collector that saves all training metrics across epochs.
    Saves to both JSON and CSV formats for easy analysis and visualization.
    """

    def __init__(self, save_dir: str, experiment_name: str = "experiment"):
        """
        Initialize metrics collector.

        Args:
            save_dir: Directory to save metrics files
            experiment_name: Name of the experiment for file naming
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.epoch_metrics = []
        self.training_start_time = None
        self.training_end_time = None

        # Metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': None,
            'end_time': None,
            'total_epochs': 0,
            'best_epoch': None,
            'best_val_recall': 0.0,
            'best_val_loss': float('inf')
        }

    def on_train_start(self, trainer, pl_module):
        """Record training start time and initial metadata."""
        self.training_start_time = datetime.now()
        self.metadata['start_time'] = self.training_start_time.isoformat()

        # Record model and training configuration
        self.metadata.update({
            'max_epochs': trainer.max_epochs,
            'num_devices': trainer.num_devices,
            'accelerator': trainer.accelerator.__class__.__name__ if trainer.accelerator else 'CPU',
            'precision': str(trainer.precision),
            'model_class': pl_module.__class__.__name__,
            'total_parameters': sum(p.numel() for p in pl_module.parameters()),
            'trainable_parameters': sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        })

        self.logger.info(f"Metrics collection started for experiment: {self.experiment_name}")

    def on_train_epoch_end(self, trainer, pl_module):
        """Collect metrics at the end of each training epoch."""
        metrics = self._extract_metrics(trainer, pl_module, 'train')

        # Store epoch metrics (will be updated with val metrics if validation runs)
        if not self.epoch_metrics or self.epoch_metrics[-1]['epoch'] != trainer.current_epoch:
            self.epoch_metrics.append(metrics)
        else:
            # Update existing entry with training metrics
            self.epoch_metrics[-1].update(metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Collect validation metrics at the end of each validation epoch."""
        if trainer.sanity_checking:
            return  # Skip sanity check metrics

        val_metrics = self._extract_metrics(trainer, pl_module, 'val')

        # Update or create epoch entry
        if self.epoch_metrics and self.epoch_metrics[-1]['epoch'] == trainer.current_epoch:
            self.epoch_metrics[-1].update(val_metrics)
        else:
            self.epoch_metrics.append(val_metrics)

        # Update best metrics
        current_val_recall = val_metrics.get('val_recall', 0.0)
        current_val_loss = val_metrics.get('val_loss', float('inf'))

        if current_val_recall > self.metadata['best_val_recall']:
            self.metadata['best_val_recall'] = current_val_recall
            self.metadata['best_epoch'] = trainer.current_epoch

        if current_val_loss < self.metadata['best_val_loss']:
            self.metadata['best_val_loss'] = current_val_loss

    def on_train_end(self, trainer, pl_module):
        """Save all collected metrics when training ends."""
        self.training_end_time = datetime.now()
        self.metadata['end_time'] = self.training_end_time.isoformat()
        self.metadata['total_epochs'] = len(self.epoch_metrics)

        if self.training_start_time:
            duration = self.training_end_time - self.training_start_time
            self.metadata['training_duration_seconds'] = duration.total_seconds()
            self.metadata['training_duration_formatted'] = str(duration)

        # Save metrics in multiple formats
        self._save_metrics()
        self.logger.info(f"Metrics saved to {self.save_dir}")

    def _extract_metrics(self, trainer, pl_module, stage: str) -> Dict[str, Any]:
        """
        Extract all available metrics from trainer.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
            stage: 'train' or 'val'

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'timestamp': datetime.now().isoformat()
        }

        # Extract all logged metrics
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            metrics[key] = value

        # Extract learning rate
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            metrics['learning_rate'] = optimizer.param_groups[0]['lr']

        # Extract from logger metrics if available
        if trainer.logged_metrics:
            for key, value in trainer.logged_metrics.items():
                if key not in metrics:
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    metrics[key] = value

        return metrics

    def _save_metrics(self):
        """Save metrics to JSON and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed metrics as JSON
        json_path = self.save_dir / f"{self.experiment_name}_metrics_{timestamp}.json"
        full_data = {
            'metadata': self.metadata,
            'epoch_metrics': self.epoch_metrics
        }

        with open(json_path, 'w') as f:
            json.dump(full_data, f, indent=2)

        self.logger.info(f"Saved JSON metrics to: {json_path}")

        # Save epoch metrics as CSV for easy plotting
        if self.epoch_metrics:
            df = pd.DataFrame(self.epoch_metrics)
            csv_path = self.save_dir / f"{self.experiment_name}_metrics_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved CSV metrics to: {csv_path}")

        # Save metadata separately
        metadata_path = self.save_dir / f"{self.experiment_name}_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        self.logger.info(f"Saved metadata to: {metadata_path}")

        # Create a summary report
        self._create_summary_report(timestamp)

    def _create_summary_report(self, timestamp: str):
        """Create a human-readable summary report."""
        report_path = self.save_dir / f"{self.experiment_name}_summary_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TRAINING SUMMARY: {self.experiment_name}\n")
            f.write("=" * 80 + "\n\n")

            f.write("EXPERIMENT METADATA:\n")
            f.write("-" * 80 + "\n")
            for key, value in self.metadata.items():
                f.write(f"{key:30s}: {value}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("EPOCH-BY-EPOCH METRICS:\n")
            f.write("=" * 80 + "\n\n")

            for epoch_data in self.epoch_metrics:
                f.write(f"\nEpoch {epoch_data.get('epoch', 'N/A')}:\n")
                f.write("-" * 80 + "\n")
                for key, value in sorted(epoch_data.items()):
                    if key not in ['epoch', 'timestamp', 'global_step']:
                        if isinstance(value, float):
                            f.write(f"  {key:28s}: {value:.6f}\n")
                        else:
                            f.write(f"  {key:28s}: {value}\n")

            # Add best metrics summary
            if self.epoch_metrics:
                f.write("\n" + "=" * 80 + "\n")
                f.write("BEST METRICS ACHIEVED:\n")
                f.write("=" * 80 + "\n")
                f.write(f"Best Validation Recall: {self.metadata['best_val_recall']:.6f} (Epoch {self.metadata['best_epoch']})\n")
                f.write(f"Best Validation Loss:   {self.metadata['best_val_loss']:.6f}\n")

        self.logger.info(f"Saved summary report to: {report_path}")

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Return the collected metrics history."""
        return self.epoch_metrics

    def get_metadata(self) -> Dict[str, Any]:
        """Return experiment metadata."""
        return self.metadata


def compute_class_weights_for_pl(train_df, class_column: str = 'Target') -> Optional[torch.Tensor]:
    """
    Compute balanced class weights for PyTorch Lightning training.

    Args:
        train_df: Training dataframe containing labels
        class_column: Column name containing class labels

    Returns:
        Tensor of class weights for binary classification, None if computation fails
    """
    try:
        labels = train_df[class_column].values
        unique_labels = np.unique(labels)

        if len(unique_labels) == 2:
            weights = class_weight.compute_class_weight(
                'balanced',
                classes=unique_labels,
                y=labels
            )
            return torch.tensor(weights, dtype=torch.float)
        else:
            logging.getLogger(__name__).warning(f"Expected 2 classes, found {len(unique_labels)}")
            return None

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to compute class weights: {e}")
        return None

def prepare_trainer_and_callbacks_pl(
    train_df_for_weights,
    class_column: str = 'Target',
    checkpoint_dir: str = 'checkpoints_pl',
    model_filename: str = 'best_model',
    constants: Optional[SystemConstants] = None,
    config: Optional[ExperimentConfig] = None,
    metrics_dir: Optional[str] = None,
    experiment_name: str = "pneumonia_detection"
) -> Dict[str, Any]:
    """
    Prepare PyTorch Lightning trainer callbacks and configuration.

    Args:
        train_df_for_weights: Training dataframe for computing class weights
        class_column: Column name containing class labels
        checkpoint_dir: Directory to save model checkpoints
        model_filename: Base filename for saved models
        constants: System constants for configuration
        config: Experiment configuration
        metrics_dir: Optional directory to save metrics (defaults to checkpoint_dir/metrics)
        experiment_name: Name of the experiment for metrics tracking

    Returns:
        Dictionary containing callbacks and trainer configuration
    """
    logger = logging.getLogger(__name__)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup default values from config or fallbacks
    patience = config.early_stopping_patience if config else 7
    min_delta = getattr(config, 'early_stopping_min_delta', 0.001)
    max_epochs = config.epochs if config else 50

    # Compute class weights
    class_weights = compute_class_weights_for_pl(train_df_for_weights, class_column)

    # ModelCheckpoint callback - save best model based on validation recall
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{model_filename}_{{epoch:02d}}_{{val_recall:.3f}}',
        monitor='val_recall',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True,

    )
    # EarlyStopping callback - stop training when validation recall stops improving
    early_stop_callback = EarlyStopping(
        monitor='val_recall',
        mode='max',
        patience=patience,
        min_delta=min_delta,
        verbose=True,
        strict=True,

    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(
        logging_interval='epoch',
        log_momentum=True,
        
    )

    # Custom highest recall tracker
    highest_recall_callback = HighestValRecallCallback()

    # Metrics collector - save all training metrics
    if metrics_dir is None:
        metrics_dir = os.path.join(checkpoint_dir, 'metrics')
    metrics_collector = MetricsCollectorCallback(
        save_dir=metrics_dir,
        experiment_name=experiment_name
    )

    # Compile callbacks list
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        lr_monitor,
        highest_recall_callback,
        metrics_collector
    ]

    # Trainer configuration
    trainer_config = {
        'callbacks': callbacks,
        'max_epochs': max_epochs,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1 if torch.cuda.is_available() else 'auto',
        'precision': '16-mixed' if torch.cuda.is_available() else 32,
        'log_every_n_steps': 50,
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'deterministic': True if constants and constants.SEED is not None else False
    }

    logger.info(f"Prepared trainer with {len(callbacks)} callbacks")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Early stopping patience: {patience}")
    logger.info(f"Class weights computed: {class_weights is not None}")

    return {
        'callbacks': callbacks,
        'trainer_config': trainer_config,
        'class_weights': class_weights,
        'checkpoint_dir': checkpoint_dir,
        'metrics_collector': metrics_collector
    }


def create_trainer_from_config(
    constants: SystemConstants,
    config: ExperimentConfig,
    callbacks: List[pl.Callback]
) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer with proper configuration.

    Args:
        constants: System constants
        config: Experiment configuration
        callbacks: List of callbacks to use

    Returns:
        Configured PyTorch Lightning trainer
    """
    # Set deterministic training if seed is provided
    if constants.SEED is not None:
        pl.seed_everything(constants.SEED, workers=True)

    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=config.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else 'auto',
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=constants.SEED is not None,
        gradient_clip_val=getattr(config, 'gradient_clip_val', 1.0),
        accumulate_grad_batches=getattr(config, 'accumulate_grad_batches', 1)
    )

    logging.getLogger(__name__).info(f"Trainer created with {len(callbacks)} callbacks")
    return trainer