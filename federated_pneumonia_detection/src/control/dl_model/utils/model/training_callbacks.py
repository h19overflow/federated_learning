"""
Training callbacks and utilities for PyTorch Lightning model training.
Provides checkpoint management, early stopping, and monitoring functionality.
"""

import os
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import torch
import numpy as np
from sklearn.utils import class_weight

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

from federated_pneumonia_detection.src.control.dl_model.utils.model.metrics_collector import (
    MetricsCollectorCallback,
)
from federated_pneumonia_detection.src.control.dl_model.utils.data.websocket_metrics_sender import (
    MetricsWebSocketSender,
)


class EarlyStoppingSignalCallback(pl.Callback):
    """
    Custom callback to detect when early stopping is triggered and signal frontend via WebSocket.

    This callback monitors the EarlyStopping callback state and sends a notification when
    training stops due to early stopping conditions being met.
    """

    def __init__(self, websocket_sender: Optional[MetricsWebSocketSender] = None):
        """
        Initialize the early stopping signal callback.

        Args:
            websocket_sender: MetricsWebSocketSender instance for frontend communication
        """
        super().__init__()
        self.websocket_sender = websocket_sender
        self.logger = logging.getLogger(__name__)
        self.early_stop_callback = None
        self.max_epochs = None
        self.has_signaled = False

    def setup(self, trainer: pl.Trainer, pl_module, stage: str) -> None:
        """
        Setup the callback - find the EarlyStopping callback if present.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
            stage: Training stage ('fit', 'validate', 'test')
        """
        self.max_epochs = trainer.max_epochs

        # Find the EarlyStopping callback in trainer's callbacks
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                self.early_stop_callback = callback
                self.logger.info(
                    f"[EarlyStoppingSignal] Found EarlyStopping callback with patience={callback.patience}"
                )
                break

        if self.websocket_sender:
            self.logger.info(f"[EarlyStoppingSignal] WebSocket sender available")
        else:
            self.logger.warning("[EarlyStoppingSignal] No WebSocket sender provided!")

    def on_train_end(self, trainer: pl.Trainer, pl_module) -> None:
        """
        Check if training ended due to early stopping (instead of max epochs reached).

        Called when training ends, whether by reaching max_epochs or early stopping.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
        """
        if self.has_signaled:
            return

        if not self.early_stop_callback or not self.websocket_sender:
            return

        # If current_epoch < max_epochs, training stopped early (likely due to early stopping)
        is_early_stopped = trainer.current_epoch < trainer.max_epochs

        if is_early_stopped:
            self.logger.info(
                f"[EarlyStoppingSignal] Detected early stopping: "
                f"current_epoch={trainer.current_epoch}, max_epochs={trainer.max_epochs}"
            )
            self._signal_early_stopping(trainer)
            self.has_signaled = True

    def _signal_early_stopping(self, trainer: pl.Trainer) -> None:
        """
        Send early stopping signal to frontend via WebSocket.

        Args:
            trainer: PyTorch Lightning trainer
        """
        try:
            # Get the best metric value from callback state
            best_value = self.early_stop_callback.best_score
            if isinstance(best_value, torch.Tensor):
                best_value = best_value.item()

            # Get monitored metric name
            metric_name = self.early_stop_callback.monitor
            current_epoch = trainer.current_epoch

            self.logger.info(
                f"[EarlyStoppingSignal] 🛑 Sending early_stopping signal - "
                f"Epoch: {current_epoch}, {metric_name}={best_value:.4f}, Patience: {self.early_stop_callback.patience}"
            )

            # Send early stopping notification
            self.websocket_sender.send_early_stopping_triggered(
                epoch=current_epoch,
                best_metric_value=float(best_value) if best_value else 0.0,
                metric_name=metric_name,
                patience=self.early_stop_callback.patience,
            )

            self.logger.info(
                f"[Early Stopping Signal] ✅ Successfully signaled at epoch {current_epoch}"
            )

        except Exception as e:
            self.logger.error(
                f"[Early Stopping Signal] ❌ Failed to signal early stopping: {e}",
                exc_info=True,
            )


class HighestValRecallCallback(pl.Callback):
    """Custom callback to track highest validation recall achieved during training."""

    def __init__(self):
        super().__init__()
        self.best_recall = 0.0
        self.logger = logging.getLogger(__name__)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Track the highest validation recall."""
        current_recall = trainer.callback_metrics.get("val_recall", 0.0)
        if isinstance(current_recall, torch.Tensor):
            current_recall = current_recall.item()

        if current_recall > self.best_recall:
            self.best_recall = current_recall
            self.logger.info(f"New best validation recall: {self.best_recall:.4f}")


def compute_class_weights_for_pl(
    train_df, class_column: str = "Target"
) -> Optional[torch.Tensor]:
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
                "balanced", classes=unique_labels, y=labels
            )
            return torch.tensor(weights, dtype=torch.float)
        else:
            logging.getLogger(__name__).warning(
                f"Expected 2 classes, found {len(unique_labels)}"
            )
            return None

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to compute class weights: {e}")
        return None


def prepare_trainer_and_callbacks_pl(
    train_df_for_weights,
    class_column: str = "Target",
    checkpoint_dir: str = "checkpoints_pl",
    model_filename: str = "best_model",
    config: Optional["ConfigManager"] = None,
    metrics_dir: Optional[str] = None,
    experiment_name: str = "pneumonia_detection",
    run_id: Optional[int] = None,
    enable_db_persistence: bool = True,
    websocket_sender: Optional[MetricsWebSocketSender] = None,
    is_federated: bool = False,
    client_id: Optional[int] = None,
    round_number: int = 0,
) -> Dict[str, Any]:
    """
    Prepare PyTorch Lightning trainer callbacks and configuration.

    Args:
        train_df_for_weights: Training dataframe for computing class weights
        class_column: Column name containing class labels
        checkpoint_dir: Directory to save model checkpoints
        model_filename: Base filename for saved models
        config: ConfigManager instance
        metrics_dir: Optional directory to save metrics (defaults to checkpoint_dir/metrics)
        experiment_name: Name of the experiment for metrics tracking
        run_id: Optional database run ID for metrics persistence
        enable_db_persistence: Whether to persist metrics to database
        websocket_sender: Optional MetricsWebSocketSender instance for frontend communication
        is_federated: If True, uses local_epochs; if False, uses epochs
        client_id: Optional client ID for federated learning context
        round_number: Round number for federated learning

    Returns:
        Dictionary containing callbacks and trainer configuration
    """
    if config is None:
        from federated_pneumonia_detection.config.config_manager import ConfigManager

        config = ConfigManager()

    logger = logging.getLogger(__name__)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create default WebSocket sender if not provided
    if websocket_sender is None:
        websocket_sender = MetricsWebSocketSender(websocket_uri="ws://localhost:8765")
        logger.info("[Training Callbacks] Created default WebSocket sender")

    # Setup default values from config
    patience = config.get("experiment.early_stopping_patience", 7)
    min_delta = config.get("experiment.early_stopping_min_delta", 0.001)
    max_epochs = (
        config.get("experiment.max-epochs", 50)
        if is_federated
        else config.get("experiment.epochs", 50)
    )
    training_mode = "federated" if is_federated else "centralized"

    logger.info(
        f"[Trainer Setup] max_epochs={max_epochs}, early_stopping_patience={patience}, min_delta={min_delta}"
    )
    if is_federated and client_id is not None:
        logger.info(
            f"[Trainer Setup] Federated mode with client_id={client_id}, round={round_number}"
        )

    # Compute class weights
    class_weights = compute_class_weights_for_pl(train_df_for_weights, class_column)

    # ModelCheckpoint callback - save best model based on validation recall
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{model_filename}_{{epoch:02d}}_{{val_recall:.3f}}",
        monitor="val_recall",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True,
    )
    # EarlyStopping callback - stop training when validation recall stops improving
    early_stop_callback = EarlyStopping(
        monitor="val_recall",
        mode="max",
        patience=patience,
        min_delta=min_delta,
        verbose=True,
        strict=True,
        log_rank_zero_only=True,
    )

    logger.info(
        f"[EarlyStopping] Monitoring 'val_recall' with patience={patience}, min_delta={min_delta}"
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(
        logging_interval="epoch",
        log_momentum=True,
    )

    # Custom highest recall tracker
    highest_recall_callback = HighestValRecallCallback()

    # Metrics collector - save all training metrics
    if metrics_dir is None:
        metrics_dir = os.path.join(checkpoint_dir, "metrics")
    metrics_collector = MetricsCollectorCallback(
        save_dir=metrics_dir,
        experiment_name=experiment_name,
        run_id=run_id,
        training_mode=training_mode,
        enable_db_persistence=True,
        websocket_uri="ws://localhost:8765",
        client_id=client_id,
        round_number=round_number,
    )

    # Early stopping signal callback - notify frontend when early stopping occurs
    early_stopping_signal = EarlyStoppingSignalCallback(
        websocket_sender=websocket_sender
    )

    # Compile callbacks list
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        early_stopping_signal,  # Must come after early_stop_callback
        lr_monitor,
        highest_recall_callback,
        metrics_collector,
    ]

    # Trainer configuration
    trainer_config = {
        "callbacks": callbacks,
        "max_epochs": max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1 if torch.cuda.is_available() else "auto",
        "precision": "16-mixed" if torch.cuda.is_available() else 32,
        "log_every_n_steps": 50,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "deterministic": True if config.get("system.seed") is not None else False,
    }

    logger.info(f"Prepared trainer with {len(callbacks)} callbacks")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Early stopping patience: {patience}")
    logger.info(f"Class weights computed: {class_weights is not None}")

    return {
        "callbacks": callbacks,
        "trainer_config": trainer_config,
        "class_weights": class_weights,
        "checkpoint_dir": checkpoint_dir,
        "metrics_collector": metrics_collector,
        "early_stopping_signal": early_stopping_signal,
    }


def create_trainer_from_config(
    config: Optional["ConfigManager"],
    callbacks: List[pl.Callback],
    is_federated: bool = False,
) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer with proper configuration.

    Args:
        config: ConfigManager instance
        callbacks: List of callbacks to use
        is_federated: If True, uses local_epochs; if False, uses epochs

    Returns:
        Configured PyTorch Lightning trainer
    """
    if config is None:
        from federated_pneumonia_detection.config.config_manager import ConfigManager

        config = ConfigManager()

    # Set deterministic training if seed is provided
    seed = config.get("system.seed")
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    epochs = (
        config.get("experiment.max-epochs", 50)
        if is_federated
        else config.get("experiment.epochs", 50)
    )
    gradient_clip_val = config.get("experiment.gradient_clip_val", 1.0)
    accumulate_grad_batches = config.get("experiment.accumulate_grad_batches", 1)

    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=seed is not None,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    logging.getLogger(__name__).info(f"Trainer created with {len(callbacks)} callbacks")
    return trainer
