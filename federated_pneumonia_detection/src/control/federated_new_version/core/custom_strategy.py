"""
Custom Flower strategy that sends training and evaluation configurations to clients.

Follows Flower conventions for:
- Weighted aggregation using num_examples
- Proper metric aggregation
- Configuration passing to clients
"""

from typing import Optional, Dict, Any
from collections.abc import Iterable
from flwr.app import ArrayRecord, ConfigRecord, Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from federated_pneumonia_detection.src.control.dl_model.internals.data.websocket_metrics_sender import (
    MetricsWebSocketSender,
)


class ConfigurableFedAvg(FedAvg):
    """
    FedAvg strategy extended to include custom configurations in train/evaluate messages.
    Ensures clients receive necessary configuration parameters for training and evaluation.
    """

    def __init__(
        self,
        train_config: Optional[Dict[str, Any]] = None,
        eval_config: Optional[Dict[str, Any]] = None,
        websocket_uri: str = "ws://localhost:8765",
        run_id: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize ConfigurableFedAvg strategy.

        Args:
            train_config: Dictionary of configuration parameters for training
            eval_config: Dictionary of configuration parameters for evaluation
            websocket_uri: WebSocket URI for sending metrics
            run_id: Database run ID for persisting metrics
            **kwargs: Additional arguments passed to FedAvg

        Note:
            By default, FedAvg uses 'num_examples' key for weighted aggregation.
            This is automatically handled by the parent class.
        """
        super().__init__(**kwargs)
        self.train_config = train_config or {}
        self.eval_config = eval_config or {}
        self.ws_sender = MetricsWebSocketSender(websocket_uri)
        self.run_id = run_id
        self.total_rounds = 0

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training with custom configs.



        Args:
            server_round: Current round of federated learning
            arrays: Current global ArrayRecord (model) to send to clients
            config: Configuration to be sent to clients for training
            grid: Grid instance for node sampling and communication

        Returns:
            Iterable of messages to be sent to selected client nodes for training
        """
        # Merge custom train config into the base config
        # Filter out None values as Flower's ConfigRecord doesn't accept them
        filtered_train_config = {
            k: v for k, v in self.train_config.items() if v is not None
        }
        config.update(filtered_train_config)
        # Call parent class to configure training with updated config
        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation with custom configs.

        Args:
            server_round: Current round of federated learning
            arrays: Current global ArrayRecord (model) to send to clients
            config: Configuration to be sent to clients for evaluation
            grid: Grid instance for node sampling and communication

        Returns:
            Iterable of messages to be sent to selected client nodes for evaluation
        """
        # Merge custom eval config into the base config
        # Filter out None values as Flower's ConfigRecord doesn't accept them
        filtered_eval_config = {
            k: v for k, v in self.eval_config.items() if v is not None
        }
        config.update(filtered_eval_config)
        # Call parent class to configure evaluation with updated config
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate evaluation metrics from multiple clients and broadcast to frontend.

        Following Flower conventions:
        - Parent class (FedAvg) performs weighted averaging using 'num_examples' key
        - Clients must include 'num_examples' in their MetricRecord for proper weighting
        - The weighted average is computed as: sum(metric * num_examples) / sum(num_examples)

        Args:
            server_round: Current round number
            replies: Iterable of reply messages from clients after evaluation

        Returns:
            Aggregated MetricRecord (dict-like) or None if aggregation failed
        """
        # Convert to list for logging
        replies_list = list(replies)

        # Log client responses for debugging
        print(f"[Strategy] Aggregating evaluation from {len(replies_list)} clients")
        for i, reply in enumerate(replies_list):
            if "metrics" in reply.content:
                metrics = dict(reply.content["metrics"])
                num_examples = metrics.get(
                    "num-examples", "NOT_FOUND"
                )  # Note: hyphen not underscore!
                print(f"[Strategy] Client {i}: num-examples={num_examples}")

        # Call parent's aggregate_evaluate to get weighted aggregated metrics
        # Parent class uses 'num_examples' key by default for weighted averaging
        aggregated_metrics = super().aggregate_evaluate(server_round, replies_list)

        # Extract and broadcast metrics via WebSocket
        if aggregated_metrics:
            print(f"[Strategy] Aggregated metrics: {dict(aggregated_metrics)}")
            round_metrics = self._extract_round_metrics(aggregated_metrics)

            # Get total rounds from config (set during strategy initialization or from context)
            total_rounds = getattr(self, "total_rounds", 0)

            # Broadcast round metrics to frontend
            self.ws_sender.send_round_metrics(
                round_num=server_round, total_rounds=total_rounds, metrics=round_metrics
            )
        else:
            print(f"[Strategy] Warning: No aggregated metrics for round {server_round}")

        return aggregated_metrics

    def _extract_round_metrics(
        self, aggregated_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract standard metrics from aggregated metrics dictionary.

        Args:
            aggregated_metrics: Dictionary containing aggregated metrics from clients

        Returns:
            Dictionary with standardized metric names (loss, accuracy, precision, recall, f1, auroc)
        """
        metrics = {}

        # Map various possible metric names to standardized names
        # Order matters: check most specific names first
        metric_mappings = {
            "loss": ["loss", "test_loss", "val_loss"],
            "accuracy": [
                "test_acc",  # Check this BEFORE test_accuracy (model logs as test_acc)
                "test_accuracy",
                "val_acc",
                "val_accuracy",
                "accuracy",
            ],
            "precision": ["test_precision", "val_precision", "precision"],
            "recall": ["test_recall", "val_recall", "recall"],
            "f1": ["test_f1", "val_f1", "f1_score", "f1"],
            "auroc": ["test_auroc", "val_auroc", "auroc", "auc", "roc_auc"],
        }

        # Extract each metric if available
        for standard_name, possible_names in metric_mappings.items():
            for name in possible_names:
                if name in aggregated_metrics:
                    metrics[standard_name] = float(aggregated_metrics[name])
                    break

        return metrics

    def set_total_rounds(self, total_rounds: int) -> None:
        """
        Set the total number of rounds for progress tracking.

        Args:
            total_rounds: Total number of federated learning rounds
        """
        self.total_rounds = total_rounds
