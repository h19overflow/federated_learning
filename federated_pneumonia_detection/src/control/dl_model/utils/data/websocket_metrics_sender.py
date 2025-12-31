import asyncio
import json
import logging
import websockets
from typing import Dict, Any, Optional
from datetime import datetime


class MetricsWebSocketSender:
    """
    Simple WebSocket sender for broadcasting training metrics to frontend.

    Handles async WebSocket communication from synchronous callback contexts
    by running async code in a new event loop.
    """

    def __init__(self, websocket_uri: str = "ws://localhost:8765"):
        """
        Initialize the WebSocket metrics sender.

        Args:
            websocket_uri: WebSocket server URI (default: ws://localhost:8765)
        """
        self.websocket_uri = websocket_uri
        self.logger = logging.getLogger(__name__)

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
                **summary_data,  # Includes status, best_epoch, etc.
            }
            self.send_metrics(payload, "training_end")
        except Exception as e:
            self.logger.warning(f"Failed to send training end notification: {e}")

    def send_metrics(
        self, metrics: Dict[str, Any], metric_type: str = "epoch_end"
    ) -> None:
        """
        Send metrics to WebSocket server.

        Wraps async send in asyncio.run() to work from synchronous contexts.

        Args:
            metrics: Dictionary of metrics to send
            metric_type: Type of metrics (e.g., 'epoch_end', 'epoch_start', 'round_end')
        """
        try:
            # Prepare payload
            payload = {
                "type": metric_type,
                "timestamp": datetime.now().isoformat(),
                "data": metrics,
            }

            self.logger.debug(
                f"[WebSocketSender] Sending {metric_type} message to {self.websocket_uri}: "
                f"{json.dumps(payload, default=str)[:200]}..."
            )

            # Run async send in a new event loop
            asyncio.run(self._send_async(payload))

        except Exception as e:
            self.logger.warning(
                f"Failed to send metrics via WebSocket: {e}", exc_info=True
            )

    async def _send_async(self, payload: Dict[str, Any]) -> None:
        """
        Async method to send payload to WebSocket server.

        Args:
            payload: Dictionary to send as JSON
        """
        try:
            self.logger.debug(
                f"[WebSocketSender] Connecting to {self.websocket_uri}..."
            )
            async with websockets.connect(self.websocket_uri, ping_interval=None) as ws:
                message_json = json.dumps(payload)
                await ws.send(message_json)
                self.logger.info(
                    f"[WebSocketSender] ✅ Successfully sent {payload['type']} to WebSocket"
                )
        except Exception as e:
            self.logger.error(
                f"[WebSocketSender] ❌ WebSocket send error for {payload['type']}: {e}",
                exc_info=True,
            )

    def send_epoch_end(self, epoch: int, phase: str, metrics: Dict[str, Any]) -> None:
        """
        Send epoch end metrics.

        Args:
            epoch: Epoch number
            phase: Phase name ('train', 'val', etc.)
            metrics: Metrics dictionary
        """
        payload_data = {"epoch": epoch, "phase": phase, "metrics": metrics}
        self.send_metrics(payload_data, "epoch_end")

    def send_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """
        Send epoch start notification.

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
        """
        payload_data = {"epoch": epoch, "total_epochs": total_epochs}
        self.send_metrics(payload_data, "epoch_start")

    def send_round_end(
        self,
        round_num: int,
        total_rounds: int,
        fit_metrics: Optional[Dict[str, Any]] = None,
        eval_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send federated learning round end notification.

        Args:
            round_num: Current round number
            total_rounds: Total number of rounds
            fit_metrics: Metrics from fit phase
            eval_metrics: Metrics from evaluation phase
        """
        payload_data = {
            "round": round_num,
            "total_rounds": total_rounds,
            "fit_metrics": fit_metrics or {},
            "eval_metrics": eval_metrics or {},
        }
        self.send_metrics(payload_data, "round_end")

    def send_status(self, status: str, message: str = "") -> None:
        """
        Send training status update.

        Args:
            status: Status string ('running', 'completed', 'error')
            message: Optional status message
        """
        payload_data = {"status": status, "message": message}
        self.send_metrics(payload_data, "status")

    def send_error(
        self, error_message: str, error_type: str = "training_error"
    ) -> None:
        """
        Send error notification.

        Args:
            error_message: Description of the error
            error_type: Type of error
        """
        payload_data = {"error": error_message, "error_type": error_type}
        self.send_metrics(payload_data, "error")

    def send_early_stopping_triggered(
        self,
        epoch: int,
        best_metric_value: float,
        metric_name: str = "val_recall",
        patience: int = 7,
    ) -> None:
        """
        Send early stopping triggered notification.

        Args:
            epoch: Epoch at which early stopping was triggered
            best_metric_value: Best metric value achieved
            metric_name: Name of the monitored metric
            patience: Patience parameter used
        """
        payload_data = {
            "epoch": epoch,
            "best_metric_value": best_metric_value,
            "metric_name": metric_name,
            "patience": patience,
            "reason": f"Early stopping triggered at epoch {epoch} with best {metric_name}={best_metric_value:.4f}",
        }
        self.logger.info(
            f"[WebSocketSender] 📤 Sending early_stopping signal - "
            f"epoch={epoch}, {metric_name}={best_metric_value:.4f}, patience={patience}"
        )
        self.send_metrics(payload_data, "early_stopping")
        self.logger.info("[WebSocketSender] ✅ Early stopping signal sent")

    def send_training_mode(
        self, is_federated: bool, num_rounds: int, num_clients: int
    ) -> None:
        """
        Signal to frontend whether this is federated or centralized training.

        Args:
            is_federated: Whether training is federated
            num_rounds: Total number of federated rounds
            num_clients: Number of participating clients
        """
        payload_data = {
            "is_federated": is_federated,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
        }
        self.logger.info(
            f"[WebSocketSender] 📤 Sending training_mode signal - "
            f"is_federated={is_federated}, num_rounds={num_rounds}, num_clients={num_clients}"
        )
        self.send_metrics(payload_data, "training_mode")
        self.logger.info("[WebSocketSender] ✅ Training mode signal sent")

    def send_round_metrics(
        self, round_num: int, total_rounds: int, metrics: Dict[str, float]
    ) -> None:
        """
        Send aggregated metrics for a federated round.

        Args:
            round_num: Current round number
            total_rounds: Total number of rounds
            metrics: Dictionary of aggregated metrics (loss, accuracy, precision, recall, f1, auroc)
        """
        payload_data = {
            "round": round_num,
            "total_rounds": total_rounds,
            "metrics": metrics,
        }
        self.logger.info(
            f"[WebSocketSender] 📤 Sending round_metrics - "
            f"round={round_num}/{total_rounds}, metrics={metrics}"
        )
        self.send_metrics(payload_data, "round_metrics")
        self.logger.info(f"[WebSocketSender] ✅ Round {round_num} metrics sent")

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
        """
        Send batch-level metrics for real-time observability.

        Args:
            step: Global training step
            batch_idx: Batch index within epoch
            loss: Batch loss value
            accuracy: Batch accuracy (if available)
            epoch: Current epoch number
            recall: Batch recall (if available)
            f1: Batch F1 score (if available)
            client_id: Optional client ID for federated learning
            round_num: Optional round number for federated learning
        """
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

        # Add federated context if applicable
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
        """
        Send gradient statistics for monitoring gradient flow.

        Args:
            step: Global training step
            total_norm: Sum of all gradient norms
            layer_norms: Dictionary mapping layer names to gradient norms
            max_norm: Maximum gradient norm across layers
            min_norm: Minimum gradient norm across layers
        """
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
        """
        Send learning rate update for LR schedule visualization.

        Args:
            current_lr: Current learning rate value
            step: Global training step
            epoch: Current epoch number
            scheduler_type: Optional name of the LR scheduler being used
        """
        payload_data = {
            "current_lr": current_lr,
            "step": step,
            "epoch": epoch,
        }

        if scheduler_type is not None:
            payload_data["scheduler_type"] = scheduler_type

        self.send_metrics(payload_data, "lr_update")
