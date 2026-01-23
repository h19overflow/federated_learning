"""
Custom Flower strategy that sends training and evaluation configurations to clients.

Follows Flower conventions for:
- Weighted aggregation using num_examples
- Proper metric aggregation
- Configuration passing to clients
"""

from collections.abc import Iterable
from typing import Any, Dict, Optional

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
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
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
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
        filtered_train_config = {
            k: v for k, v in self.train_config.items() if v is not None
        }
        config.update(filtered_train_config)
        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
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
        filtered_eval_config = {
            k: v for k, v in self.eval_config.items() if v is not None
        }
        config.update(filtered_eval_config)
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in train replies."""
        replies_list = list(replies)
        print(
            f"[Strategy] Round {server_round}: Aggregating TRAIN from {len(replies_list)} clients",
        )

        for i, reply in enumerate(replies_list):
            if "metrics" not in reply.content:
                print(
                    f"[Strategy]   Client {i}: WARNING - No 'metrics' in reply.content",
                )
                continue

            metrics = dict(reply.content["metrics"])
            num_examples = metrics.get("num-examples", "NOT_FOUND")
            print(
                f"[Strategy]   Client {i}: num-examples={num_examples} (type={type(num_examples).__name__})",
            )
            print(f"[Strategy]   Client {i}: metric_keys={sorted(metrics.keys())}")

        try:
            arrays, metrics = super().aggregate_train(server_round, replies_list)
            if arrays is None:
                print(
                    f"[Strategy] Round {server_round}: TRAIN aggregation returned arrays=None",
                )
            if metrics is None:
                print(
                    f"[Strategy] Round {server_round}: TRAIN aggregation returned metrics=None",
                )
            if metrics is not None:
                print(
                    f"[Strategy] Round {server_round}: TRAIN aggregated metric_keys={sorted(dict(metrics).keys())}",
                )
            return arrays, metrics
        except Exception as e:
            print(f"[Strategy] Round {server_round}: TRAIN AGGREGATION FAILED")
            print(f"[Strategy]   Error type: {type(e).__name__}")
            print(f"[Strategy]   Error message: {e}")
            for i, reply in enumerate(replies_list):
                if "metrics" in reply.content:
                    print(
                        f"[Strategy]   Client {i} metrics: {dict(reply.content['metrics'])}",
                    )
            raise

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[Dict[str, Any]]:
        """Aggregate evaluation metrics from multiple clients and broadcast to frontend.

        Following Flower conventions:
        - Parent class (FedAvg) performs weighted averaging using 'num_examples' key
        - Clients must include 'num_examples' in their MetricRecord for proper weighting
        """
        replies_list = list(replies)

        print(f"[Strategy] Aggregating evaluation from {len(replies_list)} clients")
        for i, reply in enumerate(replies_list):
            if "metrics" in reply.content:
                metrics = dict(reply.content["metrics"])
                num_examples = metrics.get("num-examples", "NOT_FOUND")
                print(f"[Strategy] Client {i}: num-examples={num_examples}")

        aggregated_metrics = super().aggregate_evaluate(server_round, replies_list)

        if aggregated_metrics:
            print(f"[Strategy] Aggregated metrics: {dict(aggregated_metrics)}")
            round_metrics = self._extract_round_metrics(aggregated_metrics)

            total_rounds = getattr(self, "total_rounds", 0)

            self.ws_sender.send_round_metrics(
                round_num=server_round,
                total_rounds=total_rounds,
                metrics=round_metrics,
            )

            # Persist client-aggregated metrics to database
            self._persist_aggregated_metrics(server_round, round_metrics)
        else:
            print(f"[Strategy] Warning: No aggregated metrics for round {server_round}")

        return aggregated_metrics

    def _extract_round_metrics(
        self,
        aggregated_metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """Extract standard metrics from aggregated metrics dictionary."""
        metrics = {}

        metric_mappings = {
            "loss": ["loss", "test_loss", "val_loss"],
            "accuracy": [
                "test_acc",
                "test_accuracy",
                "val_acc",
                "val_accuracy",
                "accuracy",
            ],
            "precision": ["test_precision", "val_precision", "precision"],
            "recall": ["test_recall", "val_recall", "recall"],
            "f1": ["test_f1", "val_f1", "f1_score", "f1"],
            "auroc": ["test_auroc", "val_auroc", "auroc", "auc", "roc_auc"],
            "cm_tp": ["test_cm_tp", "val_cm_tp", "cm_tp"],
            "cm_tn": ["test_cm_tn", "val_cm_tn", "cm_tn"],
            "cm_fp": ["test_cm_fp", "val_cm_fp", "cm_fp"],
            "cm_fn": ["test_cm_fn", "val_cm_fn", "cm_fn"],
        }

        for standard_name, possible_names in metric_mappings.items():
            for name in possible_names:
                if name in aggregated_metrics:
                    metrics[standard_name] = float(aggregated_metrics[name])
                    break

        return metrics

    def _persist_aggregated_metrics(
        self,
        server_round: int,
        round_metrics: Dict[str, float],
    ) -> None:
        """Persist client-aggregated metrics to RunMetric table for audit/comparison.

        Args:
            server_round: Current federated learning round number
            round_metrics: Aggregated metrics from all clients
        """
        if not hasattr(self, "run_id") or self.run_id is None:
            print(
                f"[Strategy] Warning: No run_id set, skipping aggregated metrics persistence"
            )
            return

        try:
            from federated_pneumonia_detection.src.boundary.engine import get_session
            from federated_pneumonia_detection.src.boundary.CRUD.run_metric import (
                run_metric_crud,
            )

            db = get_session()

            # Map strategy metric keys to API-compatible RunMetric names
            metric_mapping = {
                "loss": "val_loss",
                "accuracy": "val_accuracy",
                "precision": "val_precision",
                "recall": "val_recall",
                "f1": "val_f1",
                "auroc": "val_auroc",
                "cm_tp": "val_cm_tp",
                "cm_tn": "val_cm_tn",
                "cm_fp": "val_cm_fp",
                "cm_fn": "val_cm_fn",
            }

            # Build list of RunMetric entries
            metrics_to_persist = []
            for strategy_key, api_key in metric_mapping.items():
                if strategy_key in round_metrics:
                    metrics_to_persist.append(
                        {
                            "run_id": self.run_id,
                            "step": server_round,
                            "context": "aggregated",
                            "metric_name": api_key,
                            "metric_value": float(round_metrics[strategy_key]),
                        }
                    )

            if metrics_to_persist:
                run_metric_crud.bulk_create(db, metrics_to_persist)
                db.commit()
                print(
                    f"[Strategy] Persisted {len(metrics_to_persist)} aggregated metrics for round {server_round}"
                )

            db.close()

        except Exception as e:
            print(f"[Strategy] Error persisting aggregated metrics: {e}")
            # Don't raise - persistence failure shouldn't break training

    def set_total_rounds(self, total_rounds: int) -> None:
        """Set the total number of rounds for progress tracking."""
        self.total_rounds = total_rounds
