import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import pytorch_lightning as pl
from sqlalchemy.orm import Session
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.control.dl_model.utils.data.metrics_file_persister import MetricsFilePersister
from federated_pneumonia_detection.src.control.dl_model.utils.data.websocket_metrics_sender import MetricsWebSocketSender

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
        enable_db_persistence: bool = True,
        websocket_uri: Optional[str] = "ws://localhost:8765",
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
            websocket_uri: Optional WebSocket URI for real-time metrics streaming   
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.training_mode = training_mode
        self.enable_db_persistence = enable_db_persistence
        self.logger = logging.getLogger(__name__)

        # Initialize file persister
        self.file_persister = MetricsFilePersister(save_dir, experiment_name)
        
        # Initialize WebSocket sender if URI provided
        self.ws_sender = None
        if websocket_uri:
            self.ws_sender = MetricsWebSocketSender(websocket_uri)

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
        """Record training start time and create/get run in database."""
        self.training_start_time = datetime.now()
        self.metadata['start_time'] = self.training_start_time.isoformat()

        # Create or get run_id from database immediately
        if self.enable_db_persistence:
            try:
                db = get_session()
                self.run_id = self._ensure_run_exists(db)
                db.commit()  # COMMIT the transaction so other sessions can see it
                db.close()
                self.logger.info(f"Training run created: run_id={self.run_id}")
            except Exception as e:
                self.logger.error(f"Failed to create run: {e}")
                db.rollback()
                db.close()
                self.run_id = None

        # Send training start to frontend with run_id
        if self.ws_sender and self.run_id:
            self.ws_sender.send_metrics({
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "max_epochs": trainer.max_epochs,
                "training_mode": self.training_mode
            }, "training_start")

        # Record model and training configuration
        self.metadata.update({
            'run_id': self.run_id,
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

        # Send metrics to frontend via WebSocket
        if self.ws_sender:
            self.ws_sender.send_epoch_end(
                epoch=trainer.current_epoch,
                phase='val',
                metrics=val_metrics
            )

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

        # Send training completion to frontend with run_id and summary
        if self.ws_sender and self.run_id:
            self.ws_sender.send_metrics({
                "run_id": self.run_id,
                "status": "completed",
                "experiment_name": self.experiment_name,
                "best_epoch": self.metadata.get('best_epoch'),
                "best_val_recall": self.metadata.get('best_val_recall'),
                "total_epochs": len(self.epoch_metrics),
                "training_duration": self.metadata.get('training_duration_formatted')
            }, "training_end")
            self.logger.info(f"Training complete notification sent (run_id={self.run_id})")

    

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

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Return the collected metrics history."""
        return self.epoch_metrics

    def get_metadata(self) -> Dict[str, Any]:
        """Return experiment metadata."""
        return self.metadata

    def send_round_end_metrics(
        self,
        round_num: int,
        total_rounds: int,
        fit_metrics: Optional[Dict[str, Any]] = None,
        eval_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send federated learning round end metrics to WebSocket.
        
        Useful for federated learning scenarios.

        Args:
            round_num: Current round number
            total_rounds: Total number of rounds
            fit_metrics: Metrics from fit phase
            eval_metrics: Metrics from evaluation phase
        """
        if self.ws_sender:
            self.ws_sender.send_round_end(
                round_num=round_num,
                total_rounds=total_rounds,
                fit_metrics=fit_metrics,
                eval_metrics=eval_metrics
            )

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
                self.logger.debug(f"Using existing run_id={self.run_id}")
                return self.run_id
            else:
                self.logger.warning(
                    f"run_id={self.run_id} not found in database. "
                    "This may indicate a session/commit issue. Creating new run."
                )

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
            
            # Delegate metric persistence to CRUD layer
            run_crud.persist_metrics(db, run_id, self.epoch_metrics)

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
        # Delegate file persistence to the file persister
        self.file_persister.save_metrics(self.epoch_metrics, self.metadata)

        if self.enable_db_persistence and self.run_id:
            try:
                self.persist_to_database()
            except Exception as e:
                self.logger.error(f"Database persistence failed: {e}")