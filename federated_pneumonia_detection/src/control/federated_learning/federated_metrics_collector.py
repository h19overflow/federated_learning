"""
Federated learning metrics collector for tracking client-side training metrics.

This module provides comprehensive metrics collection for federated learning clients,
tracking metrics across federated rounds, local epochs, and aggregating results.

Dependencies:
- json, pandas: For metrics persistence
- datetime, pathlib: For file management

Role in System:
- Tracks per-round metrics for each federated learning client
- Records local training and evaluation metrics
- Aggregates client-level performance across rounds
- Persists metrics to JSON/CSV for analysis
- Streams metrics to frontend via WebSocket for real-time monitoring
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run_metric import run_metric_crud
from federated_pneumonia_detection.src.control.dl_model.utils.data.websocket_metrics_sender import (
    MetricsWebSocketSender,
)


class FederatedMetricsCollector:
    """
    Comprehensive metrics collector for federated learning clients.

    Tracks metrics across federated rounds and local training epochs,
    saves to JSON and CSV for analysis and visualization.
    """

    def __init__(
        self,
        save_dir: str,
        client_id: str,
        experiment_name: str = "federated_experiment",
        client_db_id: Optional[int] = None,
        run_id: Optional[int] = None,
        enable_db_persistence: bool = True,
        enable_progress_logging: bool = True,
        websocket_uri: Optional[str] = "ws://localhost:8765",
    ):
        """
        Initialize federated metrics collector.

        Args:
            save_dir: Directory to save metrics files
            client_id: Unique identifier for this client
            experiment_name: Name of the federated experiment
            client_db_id: Optional database client ID for persistence
            run_id: Optional database run ID for persistence
            enable_db_persistence: Whether to save metrics to database
            enable_progress_logging: Whether to enable real-time progress logging
            websocket_uri: Optional WebSocket URI for real-time metrics streaming
        """
        self.save_dir = Path(save_dir)
        self.client_id = client_id
        self.client_db_id = client_db_id
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.current_round_db_id = None  # Set by FlowerClient when round is created
        self.enable_db_persistence = enable_db_persistence
        self.enable_progress_logging = enable_progress_logging
        self.logger = logging.getLogger(__name__)

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize WebSocket sender for real-time metrics streaming
        self.ws_sender = None
        if websocket_uri:
            try:
                self.ws_sender = MetricsWebSocketSender(websocket_uri)
                self.logger.info(
                    f"[FederatedMetrics] WebSocket sender initialized for client {client_id}"
                )
            except Exception as e:
                self.logger.warning(
                    f"[FederatedMetrics] Failed to initialize WebSocket sender: {e}"
                )
                self.ws_sender = None

        # Metrics storage
        self.round_metrics = []  # One entry per federated round
        self.local_epoch_metrics = []  # Detailed local training metrics
        self.training_start_time = None
        self.training_end_time = None
        
        # Track round_db_id for each round
        self.round_db_id_map = {}  # Maps round_num -> round_db_id

        # Metadata
        self.metadata = {
            "experiment_name": experiment_name,
            "client_id": client_id,
            "start_time": None,
            "end_time": None,
            "total_rounds": 0,
            "total_local_epochs": 0,
            "best_round": None,
            "best_val_accuracy": 0.0,
            "best_val_loss": float("inf"),
            "total_samples_trained": 0,
        }

        self.logger.info(
            f"Federated metrics collector initialized for client {client_id}"
        )

    def set_round_db_id(self, round_db_id: Optional[int]):
        """
        Set the database round ID for the current round.

        Args:
            round_db_id: Database round ID
        """
        self.current_round_db_id = round_db_id
        
        # Also store in map if we have active round metrics
        if self.round_metrics:
            current_round_num = self.round_metrics[-1].get("round")
            if current_round_num is not None:
                self.round_db_id_map[current_round_num] = round_db_id
                self.logger.debug(
                    f"Mapped round {current_round_num} -> round_db_id={round_db_id} for client {self.client_id}"
                )
        
        self.logger.debug(f"Set round_db_id={round_db_id} for client {self.client_id}")

    def start_training(self, model_info: Optional[Dict[str, Any]] = None):
        """
        Record training start time and model information.

        Args:
            model_info: Optional dictionary with model metadata
        """
        self.training_start_time = datetime.now()
        self.metadata["start_time"] = self.training_start_time.isoformat()

        if model_info:
            self.metadata.update(model_info)

        self.logger.info(f"Training started for client {self.client_id}")

    def record_round_start(self, round_num: int, server_config: Dict[str, Any]):
        """
        Record the start of a federated round.

        Args:
            round_num: Current federated round number
            server_config: Configuration received from server
        """
        round_data = {
            "round": round_num,
            "start_time": datetime.now().isoformat(),
            "server_config": server_config,
            "local_epochs": [],
            "fit_metrics": {},
            "eval_metrics": {},
        }

        self.round_metrics.append(round_data)

        # Send round start notification via WebSocket
        if self.ws_sender:
            try:
                total_rounds = server_config.get("num_rounds", round_num + 1)
                local_epochs = server_config.get("local_epochs", 1)
                self.ws_sender.send_metrics(
                    {
                        "run_id": self.run_id,
                        "round": round_num + 1,  # 1-indexed for display
                        "round_index": round_num,  # Keep 0-indexed for internal tracking
                        "total_rounds": total_rounds,
                        "client_id": self.client_id,
                        "experiment_name": self.experiment_name,
                        "local_epochs": local_epochs,
                        "status": "starting_round",
                        "timestamp": datetime.now().isoformat(),
                    },
                    "round_start",
                )
                self.logger.info(
                    f"[Client {self.client_id}] Sent round_start event for round {round_num + 1}/{total_rounds} with {local_epochs} local epochs"
                )
            except Exception as e:
                self.logger.warning(f"Failed to send round_start via WebSocket: {e}")

        self.logger.info(
            f"Round {round_num} started for client {self.client_id}, local_epochs={server_config.get('local_epochs', 1)}"
        )

    def record_local_epoch(
        self,
        round_num: int,
        local_epoch: int,
        train_loss: float,
        learning_rate: float,
        num_samples: int,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Record metrics for a single local training epoch.

        Args:
            round_num: Current federated round
            local_epoch: Local epoch number within the round
            train_loss: Training loss for this epoch
            learning_rate: Learning rate used
            num_samples: Number of samples processed
            additional_metrics: Optional additional metrics
        """
        epoch_data = {
            "round": round_num,
            "local_epoch": local_epoch,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "num_samples": num_samples,
            "timestamp": datetime.now().isoformat(),
        }

        if additional_metrics:
            epoch_data.update(additional_metrics)

        self.local_epoch_metrics.append(epoch_data)

        # Update current round's local epochs
        if self.round_metrics and self.round_metrics[-1]["round"] == round_num:
            self.round_metrics[-1]["local_epochs"].append(epoch_data)

        # Send local epoch progress via WebSocket (optional - can be disabled for performance)
        if self.ws_sender and self.enable_progress_logging:
            try:
                metrics = {
                    "train_loss": train_loss,
                    "learning_rate": learning_rate,
                    "num_samples": num_samples,
                }
                if additional_metrics:
                    metrics.update(additional_metrics)

                self.ws_sender.send_metrics(
                    {
                        "run_id": self.run_id,
                        "round": round_num + 1,  # 1-indexed for display
                        "round_index": round_num,  # Keep 0-indexed for internal tracking
                        "client_id": self.client_id,
                        "local_epoch": local_epoch,
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat(),
                        "status": "training_progress",
                        "experiment_name": self.experiment_name,
                    },
                    "client_progress",
                )
                self.logger.info(
                    f"[Client {self.client_id}] Round {round_num + 1}, Local Epoch {local_epoch}: "
                    f"Loss={train_loss:.4f}, Samples={num_samples}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to send client_progress via WebSocket: {e}"
                )

    def record_fit_metrics(
        self,
        round_num: int,
        train_loss: float,
        num_samples: int,
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Record metrics after local training (fit) completes.

        Args:
            round_num: Current federated round
            train_loss: Final training loss
            num_samples: Number of training samples
            additional_metrics: Optional additional metrics
        """
        fit_data = {
            "train_loss": train_loss,
            "num_samples": num_samples,
            "timestamp": datetime.now().isoformat(),
        }

        if additional_metrics:
            fit_data.update(additional_metrics)

        # Update current round
        if self.round_metrics and self.round_metrics[-1]["round"] == round_num:
            self.round_metrics[-1]["fit_metrics"] = fit_data
            self.metadata["total_samples_trained"] += num_samples

        self.logger.info(
            f"Round {round_num}: Fit completed - Loss: {train_loss:.4f}, "
            f"Samples: {num_samples}"
        )

    def record_eval_metrics(
        self,
        round_num: int,
        val_loss: float,
        val_accuracy: float,
        num_samples: int,
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Record evaluation metrics after model evaluation.

        Args:
            round_num: Current federated round
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            num_samples: Number of validation samples
            additional_metrics: Optional additional metrics (precision, recall, etc.)
        """
        eval_data = {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "num_samples": num_samples,
            "timestamp": datetime.now().isoformat(),
        }

        if additional_metrics:
            eval_data.update(additional_metrics)

        # Update current round
        if self.round_metrics and self.round_metrics[-1]["round"] == round_num:
            self.round_metrics[-1]["eval_metrics"] = eval_data
            self.round_metrics[-1]["end_time"] = datetime.now().isoformat()

            # Send round end notification via WebSocket with both fit and eval metrics
            if self.ws_sender:
                try:
                    fit_metrics = self.round_metrics[-1].get("fit_metrics", {})
                    self.ws_sender.send_round_end(
                        round_num=round_num,
                        total_rounds=self.metadata.get("total_rounds", round_num + 1),
                        fit_metrics=fit_metrics,
                        eval_metrics=eval_data,
                    )
                    self.logger.debug(
                        f"[Client {self.client_id}] Sent round_end event for round {round_num}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to send round_end via WebSocket: {e}")

        # Update best metrics
        if val_accuracy > self.metadata["best_val_accuracy"]:
            self.metadata["best_val_accuracy"] = val_accuracy
            self.metadata["best_round"] = round_num

        if val_loss < self.metadata["best_val_loss"]:
            self.metadata["best_val_loss"] = val_loss

        self.logger.info(
            f"Round {round_num}: Eval completed - Loss: {val_loss:.4f}, "
            f"Accuracy: {val_accuracy:.4f}"
        )

    # TODO: Still not being recorded in the database correctly , need to cross check the model in the db ,
    #  as well as check the problem with rounds and clients being recorded only on the first round
    def _persist_metrics_to_db(self) -> bool:
        """
        Persist collected metrics to database using RunMetricCRUD.

        Returns:
            True if persistence was successful, False otherwise
        """
        if not self.run_id or not self.client_db_id:
            self.logger.warning(
                f"[Client {self.client_id}] Cannot persist metrics: "
                f"run_id={self.run_id}, client_db_id={self.client_db_id}"
            )
            return False

        try:
            db = get_session()
            metrics_persisted = 0
            metrics_failed = 0

            self.logger.info(
                f"[Client {self.client_id}] Starting metrics persistence: "
                f"run_id={self.run_id}, total_rounds={len(self.round_metrics)}"
            )

            # Persist round metrics
            for round_data in self.round_metrics:
                round_num = round_data.get("round")

                # Get round_db_id from the map
                round_db_id = self.round_db_id_map.get(round_num)
                
                # Fallback to current_round_db_id for the last round if not in map
                if round_db_id is None and round_num == len(self.round_metrics) - 1:
                    round_db_id = self.current_round_db_id

                self.logger.debug(
                    f"[Client {self.client_id}] Persisting round {round_num}: "
                    f"round_db_id={round_db_id}"
                )

                # Persist fit metrics
                fit_metrics = round_data.get("fit_metrics", {})
                if fit_metrics:
                    train_loss = fit_metrics.get("train_loss")
                    if train_loss is not None:
                        try:
                            run_metric_crud.create(
                                db,
                                run_id=self.run_id,
                                client_id=self.client_db_id,
                                round_id=round_db_id,
                                metric_name="train_loss",
                                metric_value=float(train_loss),
                                step=round_num,
                                dataset_type="train",
                                context="local",
                            )
                            metrics_persisted += 1
                            self.logger.debug(
                                f"[Client {self.client_id}] Persisted train_loss={train_loss:.4f} "
                                f"for round {round_num}"
                            )
                        except Exception as e:
                            metrics_failed += 1
                            self.logger.error(
                                f"[Client {self.client_id}] Failed to persist train_loss "
                                f"(round={round_num}, value={train_loss}): {e}",
                                exc_info=True,
                            )

                # Persist eval metrics
                eval_metrics = round_data.get("eval_metrics", {})
                if eval_metrics:
                    val_loss = eval_metrics.get("val_loss")
                    if val_loss is not None:
                        try:
                            run_metric_crud.create(
                                db,
                                run_id=self.run_id,
                                client_id=self.client_db_id,
                                round_id=round_db_id,
                                metric_name="val_loss",
                                metric_value=float(val_loss),
                                step=round_num,
                                dataset_type="val",
                                context="local",
                            )
                            metrics_persisted += 1
                            self.logger.debug(
                                f"[Client {self.client_id}] Persisted val_loss={val_loss:.4f} "
                                f"for round {round_num}"
                            )
                        except Exception as e:
                            metrics_failed += 1
                            self.logger.error(
                                f"[Client {self.client_id}] Failed to persist val_loss "
                                f"(round={round_num}, value={val_loss}): {e}",
                                exc_info=True,
                            )

                    val_accuracy = eval_metrics.get("val_accuracy")
                    if val_accuracy is not None:
                        try:
                            run_metric_crud.create(
                                db,
                                run_id=self.run_id,
                                client_id=self.client_db_id,
                                round_id=round_db_id,
                                metric_name="val_accuracy",
                                metric_value=float(val_accuracy),
                                step=round_num,
                                dataset_type="val",
                                context="local",
                            )
                            metrics_persisted += 1
                            self.logger.debug(
                                f"[Client {self.client_id}] Persisted val_accuracy={val_accuracy:.4f} "
                                f"for round {round_num}"
                            )
                        except Exception as e:
                            metrics_failed += 1
                            self.logger.error(
                                f"[Client {self.client_id}] Failed to persist val_accuracy "
                                f"(round={round_num}, value={val_accuracy}): {e}",
                                exc_info=True,
                            )

            db.commit()
            db.close()

            self.logger.info(
                f"[Client {self.client_id}] Metrics persistence complete: "
                f"persisted={metrics_persisted}, failed={metrics_failed}, "
                f"total_rounds={len(self.round_metrics)}"
            )
            return metrics_failed == 0

        except Exception as e:
            self.logger.error(
                f"[Client {self.client_id}] Critical error during metrics persistence: {e}",
                exc_info=True,
            )
            return False

    def end_training(self):
        """Record training end time, save all metrics, and persist to database."""
        self.training_end_time = datetime.now()
        self.metadata["end_time"] = self.training_end_time.isoformat()
        self.metadata["total_rounds"] = len(self.round_metrics)
        self.metadata["total_local_epochs"] = len(self.local_epoch_metrics)

        if self.training_start_time:
            duration = self.training_end_time - self.training_start_time
            self.metadata["training_duration_seconds"] = duration.total_seconds()
            self.metadata["training_duration_formatted"] = str(duration)

        # Persist metrics to database
        if self.enable_db_persistence:
            self._persist_metrics_to_db()

        # Send client training completion via WebSocket
        if self.ws_sender:
            try:
                completion_data = {
                    "run_id": self.run_id,
                    "client_id": self.client_id,
                    "client_db_id": self.client_db_id,
                    "status": "completed",
                    "total_rounds": self.metadata["total_rounds"],
                    "total_local_epochs": self.metadata["total_local_epochs"],
                    "best_round": self.metadata["best_round"],
                    "best_val_accuracy": self.metadata["best_val_accuracy"],
                    "best_val_loss": self.metadata["best_val_loss"],
                    "total_samples_trained": self.metadata["total_samples_trained"],
                    "training_duration": self.metadata.get(
                        "training_duration_formatted", "0:00:00"
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
                self.ws_sender.send_metrics(completion_data, "client_complete")
                self.logger.info(
                    f"[Client {self.client_id}] Sent client_complete event: {completion_data}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to send client_complete via WebSocket: {e}", exc_info=True
                )

    def get_round_metrics(self) -> List[Dict[str, Any]]:
        """Return round metrics history."""
        return self.round_metrics

    def get_local_epoch_metrics(self) -> List[Dict[str, Any]]:
        """Return local epoch metrics history."""
        return self.local_epoch_metrics

    def get_metadata(self) -> Dict[str, Any]:
        """Return experiment metadata."""
        return self.metadata
