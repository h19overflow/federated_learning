"""
Early stopping signal callback for detecting and signaling early stopping events.
"""

import logging
from typing import TYPE_CHECKING, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

if TYPE_CHECKING:
    from federated_pneumonia_detection.src.internals.metrics_websocket_sender import (
        MetricsWebSocketSender,
    )


class EarlyStoppingSignalCallback(pl.Callback):
    """
    Custom callback to detect when early stopping is triggered and signal frontend via WebSocket.

    This callback monitors the EarlyStopping callback state and sends a notification when
    training stops due to early stopping conditions being met.
    """

    def __init__(self, websocket_sender: Optional["MetricsWebSocketSender"] = None):
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
            self.logger.info("[EarlyStoppingSignal] WebSocket sender available")
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
                f"[Early Stopping Signal] [OK] Successfully signaled at epoch {current_epoch}"
            )

        except Exception as e:
            self.logger.error(
                f"[Early Stopping Signal] [ERROR] Failed to signal early stopping: {e}",
                exc_info=True,
            )
