"""
WebSocket metrics sender for broadcasting training metrics to the frontend.

Provides a simple interface for sending metrics with different types via WebSocket.
Handles async communication from synchronous callback contexts.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import websockets


class MetricsWebSocketSender:
    """
    WebSocket sender for broadcasting training metrics to frontend.

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

    def send_metrics(
        self,
        data: Dict[str, Any],
        metric_type: str = "epoch_end",
    ) -> None:
        """
        Send metrics to WebSocket server.

        Wraps async send in asyncio.run() to work from synchronous contexts.

        Args:
            data: Dictionary containing the metrics payload. Structure varies by metric_type.
                  Common patterns:
                  - "training_start": {"run_id", "experiment_name", "max_epochs", "training_mode"}
                  - "epoch_end": {"epoch", "phase", "metrics", "client_id?", "round_number?"}
                  - "training_end": {"run_id", "status", "best_epoch", "best_val_recall"}
                  - "round_metrics": {"round", "total_rounds", "metrics"}
                  - "batch_metrics": {"step", "batch_idx", "loss", "accuracy?", "epoch"}
                  - "gradient_stats": {"step", "total_norm", "layer_norms", "max_norm", "min_norm"}
                  - "lr_update": {"current_lr", "step", "epoch", "scheduler_type?"}
                  - "early_stopping": {"epoch", "best_metric_value", "metric_name", "patience"}
                  - "training_mode": {"is_federated", "num_rounds", "num_clients"}
                  - "status": {"status", "message"}
                  - "error": {"error", "error_type"}
            metric_type: Type of metrics being sent. Used by frontend to route/handling.
                        Common types: "epoch_end", "training_start", "training_end",
                        "round_metrics", "batch_metrics", "gradient_stats", "lr_update",
                        "early_stopping", "training_mode", "status", "error"
        """
        try:
            payload = {
                "type": metric_type,
                "timestamp": datetime.now().isoformat(),
                "data": data,
            }

            self.logger.debug(
                f"[WebSocketSender] Sending {metric_type} message to {self.websocket_uri}: "
                f"{json.dumps(payload, default=str)[:200]}..."
            )

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
            async with websockets.connect(
                self.websocket_uri, ping_interval=None
            ) as ws:
                message_json = json.dumps(payload)
                await ws.send(message_json)
                self.logger.info(
                    f"[WebSocketSender] Successfully sent {payload['type']} to WebSocket"
                )
        except Exception as e:
            self.logger.error(
                f"[WebSocketSender] WebSocket send error for {payload['type']}: {e}",
                exc_info=True,
            )
