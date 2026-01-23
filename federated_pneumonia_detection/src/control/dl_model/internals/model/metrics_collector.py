import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.control.dl_model.internals.data.metrics_file_persister import (
    MetricsFilePersister,
)
from federated_pneumonia_detection.src.control.dl_model.internals.data.websocket_metrics_sender import (
    MetricsWebSocketSender,
)


# TODO: pass the federated flag and upon completion save the metrics with the new schema that would connect to a run at least and a client
# it's already being passed as training_mode so we can just adjust the behaviour and branch based on federated or not
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
        client_id: Optional[int] = None,
        round_number: int = 0,
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
            client_id: Optional client ID for federated learning context
            round_number: Round number for federated learning
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.training_mode = training_mode
        self.enable_db_persistence = enable_db_persistence
        self.logger = logging.getLogger(__name__)

        # Federated learning context
        self.client_id = client_id
        self.db_client_id = None  # Will be set after client creation in DB
        self.federated_mode = training_mode == "federated"
        self.current_round = round_number  # Set from Flower context

        if self.federated_mode and self.client_id is not None:
            self.logger.info(
                f"[MetricsCollector] Initialized in FEDERATED mode for client_id={self.client_id}, round={self.current_round}",
            )
        else:
            self.logger.info("[MetricsCollector] Initialized in CENTRALIZED mode")

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
            "experiment_name": experiment_name,
            "start_time": None,
            "end_time": None,
            "total_epochs": 0,
            "best_epoch": None,
            "best_val_recall": 0.0,
            "best_val_loss": float("inf"),
            "training_mode": training_mode,
            "client_id": client_id,
            "round_number": round_number,
        }

    def on_train_start(self, trainer, pl_module):
        """Record training start time and create/get run and client in database."""
        self.training_start_time = datetime.now()
        self.metadata["start_time"] = self.training_start_time.isoformat()

        # Create or get run_id from database immediately
        if self.enable_db_persistence:
            try:
                db = get_session()
                self.run_id = self._ensure_run_exists(db)

                # If federated, create/get client entity
                if self.federated_mode and self.client_id is not None:
                    self.db_client_id = self._ensure_client_exists(db, self.run_id)
                    self.logger.info(
                        f"[on_train_start] Federated client created: db_client_id={self.db_client_id}",
                    )

                db.commit()  # COMMIT the transaction so other sessions can see it
                db.close()
                self.logger.info(f"Training run created: run_id={self.run_id}")
            except Exception as e:
                self.logger.error(f"Failed to create run/client: {e}")
                db.rollback()
                db.close()
                self.run_id = None
                self.db_client_id = None

        # Send training start to frontend with run_id
        if self.ws_sender and self.run_id:
            self.ws_sender.send_metrics(
                {
                    "run_id": self.run_id,
                    "experiment_name": self.experiment_name,
                    "max_epochs": trainer.max_epochs,
                    "training_mode": self.training_mode,
                    "client_id": self.client_id if self.federated_mode else None,
                },
                "training_start",
            )

        # Record model and training configuration
        self.metadata.update(
            {
                "run_id": self.run_id,
                "max_epochs": trainer.max_epochs,
                "num_devices": trainer.num_devices,
                "accelerator": trainer.accelerator.__class__.__name__
                if trainer.accelerator
                else "CPU",
                "precision": str(trainer.precision),
                "model_class": pl_module.__class__.__name__,
                "total_parameters": sum(p.numel() for p in pl_module.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in pl_module.parameters() if p.requires_grad
                ),
            },
        )

        self.logger.info(
            f"Metrics collection started for experiment: {self.experiment_name}",
        )

    def on_train_epoch_end(self, trainer, pl_module):
        """Collect metrics at the end of each training epoch."""
        metrics = self._extract_metrics(trainer, pl_module, "train")

        # Store epoch metrics (will be updated with val metrics if validation runs)
        if (
            not self.epoch_metrics
            or self.epoch_metrics[-1]["epoch"] != trainer.current_epoch
        ):
            self.epoch_metrics.append(metrics)
        else:
            # Update existing entry with training metrics
            self.epoch_metrics[-1].update(metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Collect validation metrics at the end of each validation epoch."""
        if trainer.sanity_checking:
            return  # Skip sanity check metrics

        val_metrics = self._extract_metrics(trainer, pl_module, "val")

        # Update or create epoch entry
        if (
            self.epoch_metrics
            and self.epoch_metrics[-1]["epoch"] == trainer.current_epoch
        ):
            self.epoch_metrics[-1].update(val_metrics)
        else:
            self.epoch_metrics.append(val_metrics)

        # Update best metrics
        current_val_recall = val_metrics.get("val_recall", 0.0)
        current_val_loss = val_metrics.get("val_loss", float("inf"))

        if current_val_recall > self.metadata["best_val_recall"]:
            self.metadata["best_val_recall"] = current_val_recall
            self.metadata["best_epoch"] = trainer.current_epoch

        if current_val_loss < self.metadata["best_val_loss"]:
            self.metadata["best_val_loss"] = current_val_loss

        # Send metrics to frontend via WebSocket
        if self.ws_sender:
            self.ws_sender.send_epoch_end(
                epoch=trainer.current_epoch,
                phase="val",
                metrics=val_metrics,
                client_id=self.client_id if self.federated_mode else None,
                round_num=self.current_round if self.federated_mode else None,
            )

    def on_fit_end(self, trainer, pl_module):
        """
        Save all collected metrics when fit ends (after all training and validation).

        Note: Using on_fit_end instead of on_train_end ensures that the final
        validation epoch is included in the saved metrics.

        IMPORTANT: In federated mode, clients should NOT send training_end events
        because they only complete one round of local training. The server will
        send the final training_end event when all rounds are complete.
        """
        self.training_end_time = datetime.now()
        self.metadata["end_time"] = self.training_end_time.isoformat()
        self.metadata["total_epochs"] = len(self.epoch_metrics)

        if self.training_start_time:
            duration = self.training_end_time - self.training_start_time
            self.metadata["training_duration_seconds"] = duration.total_seconds()
            self.metadata["training_duration_formatted"] = str(duration)

        # Save metrics in multiple formats
        self._save_metrics()
        self.logger.info(
            f"Metrics saved to {self.save_dir} - Total epochs: {len(self.epoch_metrics)}",
        )

        # Send training completion to frontend with run_id and summary
        # BUT: Skip this in federated mode - only server should signal completion
        if self.ws_sender and self.run_id and not self.federated_mode:
            self.ws_sender.send_metrics(
                {
                    "run_id": self.run_id,
                    "status": "completed",
                    "experiment_name": self.experiment_name,
                    "best_epoch": self.metadata.get("best_epoch"),
                    "best_val_recall": self.metadata.get("best_val_recall"),
                    "total_epochs": len(self.epoch_metrics),
                    "training_duration": self.metadata.get(
                        "training_duration_formatted",
                    ),
                },
                "training_end",
            )
            self.logger.info(
                f"Training complete notification sent (run_id={self.run_id})",
            )
        elif self.federated_mode:
            self.logger.info(
                f"[Federated Mode] Client training complete for round {self.current_round}. "
                "Server will send final training_end event when all rounds complete.",
            )

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
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "timestamp": datetime.now().isoformat(),
        }

        # Extract all logged metrics
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            metrics[key] = value

        # Extract learning rate
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            metrics["learning_rate"] = optimizer.param_groups[0]["lr"]

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
        eval_metrics: Optional[Dict[str, Any]] = None,
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
                eval_metrics=eval_metrics,
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
                    "This may indicate a session/commit issue. Creating new run.",
                )

        run_data = {
            "training_mode": self.training_mode,
            "status": "in_progress",
            "start_time": self.training_start_time or datetime.now(),
            "wandb_id": "placeholder",
            "source_path": "placeholder",
        }

        new_run = run_crud.create(db, **run_data)
        db.flush()
        self.run_id = new_run.id

        self.logger.info(
            f"Created new run with id={self.run_id} for "
            f"experiment_id={self.experiment_id}",
        )

        return self.run_id

    def _ensure_client_exists(self, db: Session, run_id: int) -> int:
        """
        Ensure client exists in database for federated learning.
        Create it if necessary.

        Args:
            db: Database session
            run_id: The run ID this client belongs to

        Returns:
            client_id: The database ID of the client entity
        """
        from federated_pneumonia_detection.src.boundary.models import Client

        client_identifier = f"client_{self.client_id}"

        # Check if client already exists for this run
        existing_client = (
            db.query(Client)
            .filter(
                Client.run_id == run_id,
                Client.client_identifier == client_identifier,
            )
            .first()
        )

        if existing_client:
            self.logger.debug(
                f"Using existing client_id={existing_client.id} for {client_identifier}",
            )
            return existing_client.id

        # Create new client entity
        try:
            new_client = Client(
                run_id=run_id,
                client_identifier=client_identifier,
                created_at=datetime.now(),
                client_config={"source_node_id": self.client_id},
            )
            db.add(new_client)
            db.flush()

            self.logger.info(
                f"[_ensure_client_exists] Created new client entity: "
                f"id={new_client.id}, identifier={client_identifier}, run_id={run_id}",
            )
            return new_client.id
        except Exception as e:
            self.logger.error(f"Failed to create client entity: {e}")
            raise

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

            # Prepare federated context if applicable
            federated_context = None
            if self.federated_mode and self.db_client_id is not None:
                federated_context = {
                    "client_id": self.db_client_id,
                    "round_number": self.current_round,
                }
                self.logger.info(
                    f"[persist_to_database] Persisting federated metrics: "
                    f"run_id={run_id}, client_id={self.db_client_id}, round={self.current_round}",
                )

            # Delegate metric persistence to CRUD layer with federated context
            run_crud.persist_metrics(
                db,
                run_id,
                self.epoch_metrics,
                federated_context=federated_context,
            )

            # Update run completion time if training finished normally
            if run_id and hasattr(self, "training_end_time") and self.training_end_time:
                try:
                    run_crud.update(
                        db,
                        run_id,
                        status="completed",
                        # end_time is set by centralized_trainer.complete_run() - not here to avoid race condition
                    )
                    self.logger.info(
                        f"Updated run {run_id} with end_time={self.training_end_time.isoformat()}",
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to update run end_time: {e}")

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
