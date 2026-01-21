"""
Batch-level metrics callback for real-time training observability.
Samples and sends batch metrics at configurable intervals.
"""

import logging
from typing import Optional
import pytorch_lightning as pl


class BatchMetricsCallback(pl.Callback):
    """
    Callback to sample and send batch-level metrics during training.

    Sends metrics every Nth batch to prevent overwhelming frontend
    with updates. Supports both centralized and federated training modes.
    """

    def __init__(
        self,
        websocket_sender,
        sample_interval: int = 10,
        client_id: Optional[int] = None,
        round_num: Optional[int] = None,
    ):
        """
        Initialize batch metrics callback.

        Args:
            websocket_sender: MetricsWebSocketSender instance for sending metrics
            sample_interval: Send metrics every N batches (default: 10)
            client_id: Optional client ID for federated learning context
            round_num: Optional round number for federated learning context
        """
        super().__init__()
        self.websocket_sender = websocket_sender
        self.sample_interval = sample_interval
        self.client_id = client_id
        self.round_num = round_num
        self.logger = logging.getLogger(__name__)
        self.batch_counter = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """
        Called when training batch ends.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
            outputs: Output from training_step
            batch: Current batch data
            batch_idx: Index of current batch
        """
        self.batch_counter += 1

        # Sample every Nth batch
        if self.batch_counter % self.sample_interval != 0:
            return

        # Extract loss from outputs
        loss = None
        if outputs is not None:
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif hasattr(outputs, "loss"):
                loss = outputs.loss

        if loss is None:
            return

        # Convert tensor to float
        loss_value = loss.item() if hasattr(loss, "item") else float(loss)

        # Extract metrics if available from logged metrics
        accuracy = None
        recall = None
        f1 = None

        if "train_acc" in trainer.callback_metrics:
            acc_tensor = trainer.callback_metrics["train_acc"]
            accuracy = (
                acc_tensor.item() if hasattr(acc_tensor, "item") else float(acc_tensor)
            )

        if "train_recall" in trainer.callback_metrics:
            recall_tensor = trainer.callback_metrics["train_recall"]
            recall = (
                recall_tensor.item()
                if hasattr(recall_tensor, "item")
                else float(recall_tensor)
            )

        if "train_f1" in trainer.callback_metrics:
            f1_tensor = trainer.callback_metrics["train_f1"]
            f1 = f1_tensor.item() if hasattr(f1_tensor, "item") else float(f1_tensor)

        # Send batch metrics via WebSocket
        if self.websocket_sender:
            self.websocket_sender.send_batch_metrics(
                step=trainer.global_step,
                batch_idx=batch_idx,
                loss=loss_value,
                accuracy=accuracy,
                recall=recall,
                f1=f1,
                epoch=trainer.current_epoch,
                client_id=self.client_id,
                round_num=self.round_num,
            )

            self.logger.debug(
                f"[BatchMetrics] Sent batch {batch_idx} metrics: "
                f"loss={loss_value:.4f}, step={trainer.global_step}"
            )
