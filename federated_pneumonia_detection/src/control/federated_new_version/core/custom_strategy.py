"""

Custom Flower strategy that sends training and evaluation configurations to clients.
"""

from typing import Iterable, Optional, Dict, Any, List, Tuple
from flwr.app import ArrayRecord, ConfigRecord, Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from federated_pneumonia_detection.src.control.dl_model.utils.data.websocket_metrics_sender import (
    MetricsWebSocketSender,
)
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud


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
        """
        super().__init__(**kwargs)
        self.train_config = train_config or {}
        self.eval_config = eval_config or {}
        self.ws_sender = MetricsWebSocketSender(websocket_uri)
        self.run_id = run_id

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
        config.update(self.train_config)
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
        config.update(self.eval_config)
        # Call parent class to configure evaluation with updated config
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[int, Dict[str, Any]]],
        failures: List[Tuple[int, Exception]],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Aggregate evaluation metrics from multiple clients and broadcast to frontend.

        Args:
            server_round: Current round number
            results: List of tuples (num_examples, metrics_dict) from clients
            failures: List of failed evaluations

        Returns:
            Tuple of (aggregated_loss, aggregated_metrics_dict)
        """
        # Call parent's aggregate_evaluate to get aggregated metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Extract and broadcast metrics via WebSocket
        if aggregated_metrics:
            round_metrics = self._extract_round_metrics(aggregated_metrics)

            # Get total rounds from config (set during strategy initialization or from context)
            total_rounds = getattr(self, "total_rounds", 0)

            # Broadcast round metrics to frontend
            self.ws_sender.send_round_metrics(
                round_num=server_round, total_rounds=total_rounds, metrics=round_metrics
            )

            # Persist round metrics to database for later retrieval
            if self.run_id:
                self._persist_round_metrics(server_round, round_metrics)

        return aggregated_loss, aggregated_metrics

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
        metric_mappings = {
            "loss": ["loss", "test_loss", "val_loss"],
            "accuracy": [
                "accuracy",
                "test_accuracy",
                "val_accuracy",
                "test_acc",
                "val_acc",
            ],
            "precision": ["precision", "test_precision", "val_precision"],
            "recall": ["recall", "test_recall", "val_recall"],
            "f1": ["f1", "test_f1", "val_f1", "f1_score"],
            "auroc": ["auroc", "test_auroc", "val_auroc", "auc", "roc_auc"],
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

    def _persist_round_metrics(self, round_num: int, metrics: Dict[str, float]) -> None:
        """
        Persist global federated round metrics to database.

        Args:
            round_num: Current round number
            metrics: Aggregated metrics for the round
        """
        db = get_session()
        try:
            # Create metrics list for this round
            round_metrics = []
            for metric_name, metric_value in metrics.items():
                round_metrics.append(
                    {
                        "epoch": round_num,  # Use round_num as 'epoch' for federated
                        f"global_{metric_name}": metric_value,  # Prefix with 'global_' to distinguish
                    }
                )

            # Persist as a single 'epoch' entry with all global metrics
            if round_metrics:
                # Flatten to single dict
                flattened_metrics = {"epoch": round_num}
                for metric_name, metric_value in metrics.items():
                    flattened_metrics[f"global_{metric_name}"] = metric_value

                run_crud.persist_metrics(
                    db=db,
                    run_id=self.run_id,
                    epoch_metrics=[flattened_metrics],
                    federated_context={"is_global": True, "round": round_num},
                )
                db.commit()
                print(f"[Strategy] Persisted global metrics for round {round_num}")
        except Exception as e:
            print(f"[Strategy] Error persisting round metrics: {e}")
            db.rollback()
        finally:
            db.close()
