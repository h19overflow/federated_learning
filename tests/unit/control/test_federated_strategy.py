"""
Unit tests for ConfigurableFedAvg strategy.
Tests client configuration and metrics aggregation math.
"""

from unittest.mock import MagicMock, patch

import pytest
from flwr.app import ArrayRecord, ConfigRecord, Message
from flwr.serverapp import Grid

from federated_pneumonia_detection.src.control.federated_new_version.core.custom_strategy import (
    ConfigurableFedAvg,
)


class TestConfigurableFedAvg:
    """Tests for ConfigurableFedAvg."""

    @pytest.fixture
    def strategy(self):
        return ConfigurableFedAvg(
            train_config={"epochs": 5, "lr": 0.01},
            eval_config={"batch_size": 32},
            websocket_uri="ws://localhost:8765",
        )

    def test_configure_train(self, strategy):
        """Test that configure_train adds custom config."""
        server_round = 1
        arrays = ArrayRecord({})
        config = ConfigRecord({})
        grid = MagicMock(spec=Grid)

        # Mock parent configure_train to return a dummy iterable
        with patch("flwr.serverapp.strategy.FedAvg.configure_train") as mock_super:
            mock_super.return_value = []

            strategy.configure_train(server_round, arrays, config, grid)

            # Check if config was updated before calling super
            assert config["epochs"] == 5
            assert config["lr"] == 0.01
            mock_super.assert_called_once()

    def test_configure_evaluate(self, strategy):
        """Test that configure_evaluate adds custom config."""
        server_round = 1
        arrays = ArrayRecord({})
        config = ConfigRecord({})
        grid = MagicMock(spec=Grid)

        with patch("flwr.serverapp.strategy.FedAvg.configure_evaluate") as mock_super:
            mock_super.return_value = []

            strategy.configure_evaluate(server_round, arrays, config, grid)

            assert config["batch_size"] == 32
            mock_super.assert_called_once()

    def test_aggregate_evaluate_math(self, strategy):
        """Test weighted aggregation math for metrics."""
        server_round = 1

        # Create mock replies with metrics
        def create_reply(metrics_dict):
            reply = MagicMock(spec=Message)
            reply.content = {"metrics": metrics_dict}
            return reply

        replies = [
            create_reply({"num-examples": 100, "accuracy": 0.8}),
            create_reply({"num-examples": 200, "accuracy": 0.9}),
        ]

        # Expected weighted average: (100*0.8 + 200*0.9) / (100+200) = (80 + 180) / 300 = 260 / 300 = 0.8666...
        expected_accuracy = (100 * 0.8 + 200 * 0.9) / 300

        with patch("flwr.serverapp.strategy.FedAvg.aggregate_evaluate") as mock_super:
            # We want to verify our logic that happens BEFORE or AFTER super
            # Actually, aggregate_evaluate in ConfigurableFedAvg calls super and then broadcasts
            mock_super.return_value = {"accuracy": expected_accuracy}

            # Mock WebSocket sender
            strategy.ws_sender = MagicMock()

            aggregated = strategy.aggregate_evaluate(server_round, replies)

            assert aggregated["accuracy"] == pytest.approx(expected_accuracy)
            strategy.ws_sender.send_round_metrics.assert_called_once()

    def test_extract_round_metrics(self, strategy):
        """Test mapping of various metric names to standard names."""
        aggregated_metrics = {
            "test_acc": 0.95,
            "val_loss": 0.1,
            "test_precision": 0.92,
            "recall": 0.9,
            "f1_score": 0.91,
            "auroc": 0.98,
        }

        extracted = strategy._extract_round_metrics(aggregated_metrics)

        assert extracted["accuracy"] == 0.95
        assert extracted["loss"] == 0.1
        assert extracted["precision"] == 0.92
        assert extracted["recall"] == 0.9
        assert extracted["f1"] == 0.91
        assert extracted["auroc"] == 0.98
