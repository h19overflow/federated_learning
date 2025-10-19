import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import pandas as pd
import json
import pytorch_lightning as pl
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud

# TODO: change the run_id to int
class MetricsCollectorCallback(pl.Callback):
    """
    Comprehensive metrics collector that saves all training metrics across epochs.
    Saves to both JSON and CSV formats for easy analysis and visualization.
    """

    def __init__(
        self,
        save_dir: str,
        experiment_name: str = "experiment",
        run_id: Optional[int] = None,
        experiment_id: Optional[int] = None,
        training_mode: str = "centralized",
        enable_db_persistence: bool = True
    ):
        """
        Initialize metrics collector.

        Args:
            save_dir: Directory to save metrics files
            experiment_name: Name of the experiment for file naming
            run_id: Optional database run ID for persistence
            experiment_id: Required for creating run if run_id doesn't exist
            training_mode: Training mode (centralized, federated, etc.)
            enable_db_persistence: Whether to save metrics to database
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.training_mode = training_mode
        self.enable_db_persistence = enable_db_persistence
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

    def _ensure_run_exists(self, db: Session) -> int:
        """
        Ensure run exists in database. Create it if necessary.
        
        Returns:
            run_id: The ID of the run
        """
        if self.run_id is not None:
            # Run already exists, verify it's in database
            existing_run = run_crud.get(db, self.run_id)
            if existing_run:
                return self.run_id
        run_data = {
            'training_mode': self.training_mode,
            'status': 'in_progress',
            'start_time': self.training_start_time or datetime.now(),
            'wandb_id': 'placeholder',
            'source_path': 'placeholder',
        }
        
        new_run = run_crud.create(db, **run_data)
        db.flush()
        self.run_id = new_run.id
        
        self.logger.info(
            f"Created new run with id={self.run_id} for "
            f"experiment_id={self.experiment_id}"
        )
        
        return self.run_id

    def persist_to_database(self, db: Optional[Session] = None):
        """
        Persist collected metrics to database.

        Args:
            db: Optional database session. If None, creates a new session.
        """
        if not self.enable_db_persistence:
            self.logger.info("Database persistence is disabled")
            return

        close_session = False
        if db is None:
            db = get_session()
            close_session = True

        try:
            # Ensure run exists before persisting metrics
            run_id = self._ensure_run_exists(db)
            
            # Persist all epoch metrics to database
            metrics_to_persist = []

            for epoch_data in self.epoch_metrics:
                epoch = epoch_data.get('epoch', 0)

                # Extract and persist each metric type
                for key, value in epoch_data.items():
                    if key in ['epoch', 'timestamp', 'global_step']:
                        continue

                    if not isinstance(value, (int, float)):
                        continue

                    # Determine dataset type from metric name
                    if key.startswith('train_'):
                        dataset_type = 'train'
                        metric_name = key
                    elif key.startswith('val_'):
                        dataset_type = 'validation'
                        metric_name = key
                    elif key.startswith('test_'):
                        dataset_type = 'test'
                        metric_name = key
                    else:
                        dataset_type = 'other'
                        metric_name = key

                    metrics_to_persist.append({
                        'run_id': run_id,
                        'metric_name': metric_name,
                        'metric_value': float(value),
                        'step': epoch,
                        'dataset_type': dataset_type
                    })

            # Bulk create metrics for efficiency
            if metrics_to_persist:
                run_metric_crud.bulk_create(db, metrics_to_persist)
                db.commit()
                self.logger.info(
                    f"Persisted {len(metrics_to_persist)} metrics to database "
                    f"for run_id={run_id}"
                )

        except Exception as e:
            self.logger.error(f"Failed to persist metrics to database: {e}")
            if close_session:
                db.rollback()
            raise
        finally:
            if close_session:
                db.close()

    def _save_metrics(self):
        """Save metrics to JSON and CSV files, and optionally to database."""
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

        # Persist to database if enabled
        if self.enable_db_persistence and self.run_id:
            try:
                self.persist_to_database()
            except Exception as e:
                self.logger.error(f"Database persistence failed: {e}")