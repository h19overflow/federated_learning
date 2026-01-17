"""
SSE Metrics Sender for Training Progress.

Publishes training metrics to SSE event queues instead of WebSocket.
Drop-in replacement for MetricsWebSocketSender.
"""
import logging
import asyncio
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from federated_pneumonia_detection.src.control.dl_model.utils.data.sse_event_manager import (
    get_sse_event_manager
)


class MetricsSSESender:
    """
    SSE-based metrics sender for training progress.

    Compatible API with MetricsWebSocketSender for easy migration.
    Publishes metrics to in-memory event queue instead of WebSocket.
    """

    def __init__(self, experiment_id: str):
        """
        Initialize SSE metrics sender.

        Args:
            experiment_id: Unique experiment identifier for routing events
        """
        self.experiment_id = experiment_id
        self.logger = logging.getLogger(__name__)
        self._event_manager = None
        self._loop = None
        self._thread = None
        self._start_event_loop()
        self.logger.info(f"MetricsSSESender initialized for: {experiment_id}")

    def _start_event_loop(self):
        """Start background event loop in a separate thread."""
        def run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=run_loop,
            args=(self._loop,),
            daemon=True,
            name=f"SSESender-{self.experiment_id}"
        )
        self._thread.start()
        self.logger.debug(f"Started background event loop for {self.experiment_id}")

    async def _get_event_manager(self):
        """Lazy-load event manager."""
        if self._event_manager is None:
            self._event_manager = await get_sse_event_manager()
        return self._event_manager

    def close(self):
        """Stop the background event loop and cleanup resources."""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self.logger.debug(f"Stopped event loop for {self.experiment_id}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

    def send_training_end(self, run_id: int, summary_data: Dict[str, Any]) -> None:
        """
        Send training end notification with run_id.

        Args:
            run_id: Database run ID for querying results
            summary_data: Summary metrics and metadata
        """
        try:
            payload = {
                "run_id": run_id,
                **summary_data,
            }
            self.send_metrics(payload, "training_end")
        except Exception as e:
            self.logger.warning(f"Failed to send training end notification: {e}")

    def send_metrics(
        self, metrics: Dict[str, Any], metric_type: str = "epoch_end"
    ) -> None:
        """
        Send metrics via SSE event queue.

        Compatible API with WebSocket sender for drop-in replacement.

        Args:
            metrics: Dictionary of metrics to send
            metric_type: Type of metrics (e.g., 'epoch_end', 'batch_metrics')
        """
        try:
            payload = {
                "type": metric_type,
                "timestamp": datetime.now().isoformat(),
                "data": metrics,
            }

            self.logger.debug(
                f"[SSESender] Publishing {metric_type} for {self.experiment_id}"
            )

            # Schedule coroutine on the background event loop
            asyncio.run_coroutine_threadsafe(
                self._publish_async(payload),
                self._loop
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to send metrics via SSE: {e}", exc_info=True
            )

    async def _publish_async(self, payload: Dict[str, Any]) -> None:
        """
        Async method to publish payload to event queue.

        Args:
            payload: Dictionary with 'type', 'timestamp', 'data' keys
        """
        try:
            event_manager = await self._get_event_manager()
            await event_manager.publish_event(self.experiment_id, payload)
            self.logger.debug(
                f"[SSESender] Successfully published {payload['type']} "
                f"to {self.experiment_id}"
            )
        except Exception as e:
            self.logger.error(
                f"[SSESender] Failed to publish {payload['type']}: {e}",
                exc_info=True,
            )

    def send_epoch_end(self, epoch: int, phase: str, metrics: Dict[str, Any]) -> None:
        """Send epoch end metrics."""
        payload_data = {"epoch": epoch, "phase": phase, "metrics": metrics}
        self.send_metrics(payload_data, "epoch_end")

    def send_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Send epoch start notification."""
        payload_data = {"epoch": epoch, "total_epochs": total_epochs}
        self.send_metrics(payload_data, "epoch_start")

    def send_round_end(
        self,
        round_num: int,
        total_rounds: int,
        fit_metrics: Optional[Dict[str, Any]] = None,
        eval_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send federated learning round end notification."""
        payload_data = {
            "round": round_num,
            "total_rounds": total_rounds,
            "fit_metrics": fit_metrics or {},
            "eval_metrics": eval_metrics or {},
        }
        self.send_metrics(payload_data, "round_end")

    def send_status(self, status: str, message: str = "") -> None:
        """Send training status update."""
        payload_data = {"status": status, "message": message}
        self.send_metrics(payload_data, "status")

    def send_error(
        self, error_message: str, error_type: str = "training_error"
    ) -> None:
        """Send error notification."""
        payload_data = {"error": error_message, "error_type": error_type}
        self.send_metrics(payload_data, "error")

    def send_early_stopping_triggered(
        self,
        epoch: int,
        best_metric_value: float,
        metric_name: str = "val_recall",
        patience: int = 7,
    ) -> None:
        """Send early stopping triggered notification."""
        payload_data = {
            "epoch": epoch,
            "best_metric_value": best_metric_value,
            "metric_name": metric_name,
            "patience": patience,
            "reason": f"Early stopping triggered at epoch {epoch} with best {metric_name}={best_metric_value:.4f}",
        }
        self.logger.info(
            f"[SSESender] Sending early_stopping signal - "
            f"epoch={epoch}, {metric_name}={best_metric_value:.4f}"
        )
        self.send_metrics(payload_data, "early_stopping")

    def send_training_mode(
        self, is_federated: bool, num_rounds: int, num_clients: int
    ) -> None:
        """Signal training mode to frontend."""
        payload_data = {
            "is_federated": is_federated,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
        }
        self.logger.info(
            f"[SSESender] Sending training_mode signal - "
            f"is_federated={is_federated}, num_rounds={num_rounds}"
        )
        self.send_metrics(payload_data, "training_mode")

    def send_round_metrics(
        self, round_num: int, total_rounds: int, metrics: Dict[str, float]
    ) -> None:
        """Send aggregated metrics for a federated round."""
        payload_data = {
            "round": round_num,
            "total_rounds": total_rounds,
            "metrics": metrics,
        }
        self.logger.info(
            f"[SSESender] Sending round_metrics - round={round_num}/{total_rounds}"
        )
        self.send_metrics(payload_data, "round_metrics")

    def send_batch_metrics(
        self,
        step: int,
        batch_idx: int,
        loss: float,
        accuracy: Optional[float],
        epoch: int,
        recall: Optional[float] = None,
        f1: Optional[float] = None,
        client_id: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> None:
        """Send batch-level metrics for real-time observability."""
        payload_data = {
            "step": step,
            "batch_idx": batch_idx,
            "loss": loss,
            "accuracy": accuracy,
            "recall": recall,
            "f1": f1,
            "epoch": epoch,
            "timestamp": datetime.now().timestamp(),
        }

        if client_id is not None:
            payload_data["client_id"] = client_id
        if round_num is not None:
            payload_data["round_num"] = round_num

        self.send_metrics(payload_data, "batch_metrics")

    def send_gradient_stats(
        self,
        step: int,
        total_norm: float,
        layer_norms: Dict[str, float],
        max_norm: float,
        min_norm: float,
    ) -> None:
        """Send gradient statistics for monitoring gradient flow."""
        payload_data = {
            "step": step,
            "total_norm": total_norm,
            "layer_norms": layer_norms,
            "max_norm": max_norm,
            "min_norm": min_norm,
        }
        self.send_metrics(payload_data, "gradient_stats")

    def send_lr_update(
        self,
        current_lr: float,
        step: int,
        epoch: int,
        scheduler_type: Optional[str] = None,
    ) -> None:
        """Send learning rate update for LR schedule visualization."""
        payload_data = {
            "current_lr": current_lr,
            "step": step,
            "epoch": epoch,
        }

        if scheduler_type is not None:
            payload_data["scheduler_type"] = scheduler_type

        self.send_metrics(payload_data, "lr_update")
