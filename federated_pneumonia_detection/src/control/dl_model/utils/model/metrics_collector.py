import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import torch
import pandas as pd
import json
import pytorch_lightning as pl


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