"""
WebSocket-enabled progress logger for real-time frontend updates.

Extends the file-based progress logger to broadcast updates via WebSocket.
Safe for multiprocessing by using a queue-based approach.

Dependencies:
- asyncio: For async WebSocket broadcasting
- queue: For thread-safe message passing
- threading: For background WebSocket broadcaster

Role in System:
- Broadcasts training progress to connected WebSocket clients
- Maintains file-based logging as backup
- Thread-safe and multiprocessing-safe
- Can be enabled/disabled without code changes
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from threading import Thread
from queue import Queue

from .progress_logger import ProgressLogger, FederatedProgressLogger


class WebSocketProgressLogger(ProgressLogger):
    """
    Progress logger with WebSocket broadcasting capability.

    Extends file-based logger to broadcast updates to WebSocket clients.
    Uses a queue-based approach to avoid pickling issues with multiprocessing.
    """

    def __init__(
        self,
        websocket_manager: Optional[Any] = None,
        experiment_id: Optional[str] = None,
        log_file: str = "training_progress.json",
        log_dir: str = "logs/progress",
        experiment_name: str = "experiment",
        mode: str = "centralized",
        enable_file_logging: bool = True,
        broadcast_interval: float = 0.1,
    ):
        """
        Initialize WebSocket-enabled progress logger.

        Args:
            websocket_manager: ConnectionManager instance for WebSocket broadcasting
            experiment_id: Experiment ID for WebSocket channel isolation
            log_file: Name of the progress log file
            log_dir: Directory to store progress logs
            experiment_name: Name of the experiment
            mode: Training mode ('centralized' or 'federated')
            enable_file_logging: Whether to also log to file (recommended as backup)
            broadcast_interval: Minimum seconds between broadcasts (prevents flooding)
        """
        # Initialize parent file logger
        super().__init__(
            log_file=log_file,
            log_dir=log_dir,
            experiment_name=experiment_name,
            mode=mode,
        )

        self.websocket_manager = websocket_manager
        self.experiment_id = experiment_id or experiment_name
        self.enable_file_logging = enable_file_logging
        self.broadcast_interval = broadcast_interval
        self._last_broadcast = 0.0

        # Message queue for thread-safe WebSocket broadcasting
        self.message_queue = Queue()
        self.broadcaster_thread = None
        self.is_running = False

        # Start broadcaster thread if WebSocket manager is provided
        if self.websocket_manager:
            self._start_broadcaster()
            self.logger.info("WebSocket broadcasting enabled")

    def _start_broadcaster(self):
        """Start background thread for WebSocket broadcasting."""
        self.is_running = True
        self.broadcaster_thread = Thread(target=self._broadcast_worker, daemon=True)
        self.broadcaster_thread.start()

    def _broadcast_worker(self):
        """Background worker that broadcasts messages from queue."""
        # Create a dedicated event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while self.is_running:
                try:
                    # Get message from queue (blocks with timeout)
                    message = self.message_queue.get(timeout=1.0)

                    # Broadcast via WebSocket using thread's event loop
                    if self.websocket_manager:
                        loop.run_until_complete(self._async_broadcast(message))

                    self.message_queue.task_done()

                except Exception as e:
                    if not isinstance(e, Exception) or "Empty" not in str(type(e)):
                        self.logger.warning(f"WebSocket broadcast error: {e}")
        finally:
            # Clean up event loop when thread exits
            loop.close()

    async def _async_broadcast(self, message: Dict[str, Any]):
        """
        Async broadcast message to all WebSocket clients.

        Args:
            message: Message to broadcast (must have 'type' field)
        """
        try:
            # Wrap message in proper structure for frontend
            # Frontend expects: { type, data, timestamp }
            message_type = message.pop("type", "unknown")

            wrapped_message = {
                "type": message_type,
                "data": message,  # Everything except 'type' goes into 'data'
                "timestamp": message.get("timestamp", datetime.now().isoformat()),
            }

            # Remove timestamp from data to avoid duplication
            wrapped_message["data"].pop("timestamp", None)

            # Broadcast dict directly (ConnectionManager will encode it)
            await self.websocket_manager.broadcast(wrapped_message, self.experiment_id)

        except Exception as e:
            self.logger.error(f"Failed to broadcast via WebSocket: {e}")

    def _should_broadcast(self) -> bool:
        """Check if enough time has passed since last broadcast."""
        current_time = datetime.now().timestamp()
        if current_time - self._last_broadcast >= self.broadcast_interval:
            self._last_broadcast = current_time
            return True
        return False

    def _enqueue_broadcast(self, message: Dict[str, Any]):
        """
        Enqueue message for WebSocket broadcasting.

        Args:
            message: Message to broadcast
        """
        if self.websocket_manager and self._should_broadcast():
            try:
                self.message_queue.put_nowait(message)
            except Exception as e:
                self.logger.warning(f"Failed to enqueue message: {e}")

    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start with WebSocket broadcast."""
        # File logging
        if self.enable_file_logging:
            super().log_epoch_start(epoch, total_epochs)

        # WebSocket broadcast
        message = {
            "type": "epoch_start",
            "epoch": epoch,
            "total_epochs": total_epochs,
            "timestamp": datetime.now().isoformat(),
        }
        self._enqueue_broadcast(message)

    def log_epoch_end(
        self, epoch: int, metrics: Dict[str, float], phase: str = "train"
    ):
        """Log epoch end with WebSocket broadcast."""
        # File logging
        if self.enable_file_logging:
            super().log_epoch_end(epoch, metrics, phase)

        # WebSocket broadcast
        message = {
            "type": "epoch_end",
            "epoch": epoch,
            "phase": phase,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self._enqueue_broadcast(message)

    def log_training_complete(self, final_metrics: Dict[str, Any]):
        """Log training completion with WebSocket broadcast."""
        # File logging
        if self.enable_file_logging:
            super().log_training_complete(final_metrics)

        # WebSocket broadcast
        message = {
            "type": "training_complete",
            "final_metrics": final_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self._enqueue_broadcast(message)

    def log_training_error(self, error_message: str):
        """Log training error with WebSocket broadcast."""
        # File logging
        if self.enable_file_logging:
            super().log_training_error(error_message)

        # WebSocket broadcast
        message = {
            "type": "training_error",
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
        }
        self._enqueue_broadcast(message)

    def log_custom_event(self, event_type: str, data: Dict[str, Any]):
        """Log custom event with WebSocket broadcast."""
        # File logging
        if self.enable_file_logging:
            super().log_custom_event(event_type, data)

        # WebSocket broadcast
        message = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        self._enqueue_broadcast(message)

    def shutdown(self):
        """Gracefully shutdown the broadcaster thread."""
        self.is_running = False
        if self.broadcaster_thread:
            self.broadcaster_thread.join(timeout=2.0)
        self.logger.info("WebSocket broadcaster shut down")


class WebSocketFederatedProgressLogger(FederatedProgressLogger):
    """
    Federated progress logger with WebSocket broadcasting.

    Extends federated file-based logger for real-time WebSocket updates.
    """

    def __init__(
        self,
        client_id: str,
        websocket_manager: Optional[Any] = None,
        experiment_id: Optional[str] = None,
        log_file: Optional[str] = None,
        log_dir: str = "logs/progress",
        experiment_name: str = "federated_experiment",
        enable_file_logging: bool = True,
        broadcast_interval: float = 0.1,
    ):
        """
        Initialize WebSocket-enabled federated progress logger.

        Args:
            client_id: Unique client identifier
            websocket_manager: ConnectionManager for WebSocket broadcasting
            experiment_id: Experiment ID for WebSocket channel isolation
            log_file: Optional custom log file name
            log_dir: Directory to store progress logs
            experiment_name: Name of the experiment
            enable_file_logging: Whether to also log to file
            broadcast_interval: Minimum seconds between broadcasts
        """
        # Initialize parent federated logger
        super().__init__(
            client_id=client_id,
            log_file=log_file,
            log_dir=log_dir,
            experiment_name=experiment_name,
        )

        self.websocket_manager = websocket_manager
        self.experiment_id = experiment_id or experiment_name
        self.enable_file_logging = enable_file_logging
        self.broadcast_interval = broadcast_interval
        self._last_broadcast = 0.0

        # Message queue for thread-safe broadcasting
        self.message_queue = Queue()
        self.broadcaster_thread = None
        self.is_running = False

        # Start broadcaster if WebSocket manager provided
        if self.websocket_manager:
            self._start_broadcaster()
            self.logger.info(f"WebSocket broadcasting enabled for client {client_id}")

    def _start_broadcaster(self):
        """Start background broadcaster thread."""
        self.is_running = True
        self.broadcaster_thread = Thread(target=self._broadcast_worker, daemon=True)
        self.broadcaster_thread.start()

    def _broadcast_worker(self):
        """Background worker for WebSocket broadcasting."""
        # Create a dedicated event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while self.is_running:
                try:
                    message = self.message_queue.get(timeout=1.0)

                    if self.websocket_manager:
                        loop.run_until_complete(self._async_broadcast(message))

                    self.message_queue.task_done()

                except Exception as e:
                    if not isinstance(e, Exception) or "Empty" not in str(type(e)):
                        self.logger.warning(f"WebSocket broadcast error: {e}")
        finally:
            # Clean up event loop when thread exits
            loop.close()

    async def _async_broadcast(self, message: Dict[str, Any]):
        """Async broadcast to WebSocket clients."""
        try:
            # Wrap message in proper structure for frontend
            message_type = message.pop("type", "unknown")

            wrapped_message = {
                "type": message_type,
                "data": message,
                "timestamp": message.get("timestamp", datetime.now().isoformat()),
            }

            # Remove timestamp from data to avoid duplication
            wrapped_message["data"].pop("timestamp", None)

            # Broadcast dict directly (ConnectionManager will encode it)
            await self.websocket_manager.broadcast(wrapped_message, self.experiment_id)
        except Exception as e:
            self.logger.error(f"WebSocket broadcast failed: {e}")

    def _should_broadcast(self) -> bool:
        """Check if broadcast throttling allows sending."""
        current_time = datetime.now().timestamp()
        if current_time - self._last_broadcast >= self.broadcast_interval:
            self._last_broadcast = current_time
            return True
        return False

    def _enqueue_broadcast(self, message: Dict[str, Any]):
        """Enqueue message for broadcasting."""
        if self.websocket_manager and self._should_broadcast():
            try:
                self.message_queue.put_nowait(message)
            except Exception as e:
                self.logger.warning(f"Failed to enqueue: {e}")

    def log_round_start(self, round_num: int, total_rounds: int):
        """Log round start with WebSocket broadcast."""
        if self.enable_file_logging:
            super().log_round_start(round_num, total_rounds)

        message = {
            "type": "round_start",
            "round": round_num,
            "total_rounds": total_rounds,
            "client_id": self.client_id,
            "timestamp": datetime.now().isoformat(),
        }
        self._enqueue_broadcast(message)

    def log_round_end(
        self,
        round_num: int,
        fit_metrics: Dict[str, float],
        eval_metrics: Dict[str, float],
    ):
        """Log round end with WebSocket broadcast."""
        if self.enable_file_logging:
            super().log_round_end(round_num, fit_metrics, eval_metrics)

        message = {
            "type": "round_end",
            "round": round_num,
            "client_id": self.client_id,
            "fit_metrics": fit_metrics,
            "eval_metrics": eval_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self._enqueue_broadcast(message)

    def log_local_epoch(
        self, round_num: int, local_epoch: int, metrics: Dict[str, float]
    ):
        """Log local epoch with WebSocket broadcast."""
        if self.enable_file_logging:
            super().log_local_epoch(round_num, local_epoch, metrics)

        message = {
            "type": "local_epoch",
            "round": round_num,
            "local_epoch": local_epoch,
            "client_id": self.client_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self._enqueue_broadcast(message)

    def shutdown(self):
        """Gracefully shutdown broadcaster."""
        self.is_running = False
        if self.broadcaster_thread:
            self.broadcaster_thread.join(timeout=2.0)
        self.logger.info(f"WebSocket broadcaster shut down for client {self.client_id}")
