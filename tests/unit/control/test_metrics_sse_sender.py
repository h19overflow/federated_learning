"""
Unit tests for SSE Metrics Sender.

Tests sender initialization, metric sending, and compatibility methods.
Uses synchronous API - no asyncio needed, compatible with Ray actors.
"""
import pytest
from unittest.mock import MagicMock, patch
from federated_pneumonia_detection.src.control.dl_model.utils.data.metrics_sse_sender import (
    MetricsSSESender
)
from federated_pneumonia_detection.src.control.dl_model.utils.data.sse_event_manager import (
    SSEEventManager
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instance before each test."""
    SSEEventManager._instance = None
    import federated_pneumonia_detection.src.control.dl_model.utils.data.sse_event_manager as mod
    mod._manager_instance = None
    yield


def test_sender_initialization():
    """Test that SSE sender initializes correctly."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_1")

    assert sender.experiment_id == "test_exp_sender_1"
    assert sender._event_manager is not None  # Eagerly loaded now


def test_send_metrics_basic():
    """Test that metrics are sent via event manager (synchronous)."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_2")

    # Mock event manager
    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    # Send metrics
    sender.send_metrics({"loss": 0.5}, "epoch_end")

    # Verify publish_event was called (synchronously, no sleep needed)
    mock_manager.publish_event.assert_called_once()
    call_args = mock_manager.publish_event.call_args[0]
    assert call_args[0] == "test_exp_sender_2"
    assert call_args[1]["type"] == "epoch_end"
    assert call_args[1]["data"]["loss"] == 0.5
    assert "timestamp" in call_args[1]


def test_send_epoch_end():
    """Test epoch_end convenience method."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_3")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    sender.send_epoch_end(epoch=5, phase="train", metrics={"loss": 0.3, "accuracy": 0.85})

    mock_manager.publish_event.assert_called_once()
    call_args = mock_manager.publish_event.call_args[0]
    assert call_args[1]["type"] == "epoch_end"
    assert call_args[1]["data"]["epoch"] == 5
    assert call_args[1]["data"]["phase"] == "train"
    assert call_args[1]["data"]["metrics"]["loss"] == 0.3
    assert call_args[1]["data"]["metrics"]["accuracy"] == 0.85


def test_send_epoch_start():
    """Test epoch_start convenience method."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_4")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    sender.send_epoch_start(epoch=1, total_epochs=10)

    mock_manager.publish_event.assert_called_once()
    call_args = mock_manager.publish_event.call_args[0]
    assert call_args[1]["type"] == "epoch_start"
    assert call_args[1]["data"]["epoch"] == 1
    assert call_args[1]["data"]["total_epochs"] == 10


def test_send_batch_metrics():
    """Test batch_metrics method with all parameters."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_5")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    sender.send_batch_metrics(
        step=42,
        batch_idx=10,
        loss=0.563,
        accuracy=0.812,
        recall=0.756,
        f1=0.783,
        epoch=2,
        client_id=1,
        round_num=3
    )

    mock_manager.publish_event.assert_called_once()
    call_args = mock_manager.publish_event.call_args[0]
    payload = call_args[1]["data"]

    assert call_args[1]["type"] == "batch_metrics"
    assert payload["step"] == 42
    assert payload["batch_idx"] == 10
    assert payload["loss"] == 0.563
    assert payload["accuracy"] == 0.812
    assert payload["recall"] == 0.756
    assert payload["f1"] == 0.783
    assert payload["epoch"] == 2
    assert payload["client_id"] == 1
    assert payload["round_num"] == 3


def test_send_status():
    """Test status message sending."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_6")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    sender.send_status(status="running", message="Training in progress")

    call_args = mock_manager.publish_event.call_args[0]
    assert call_args[1]["type"] == "status"
    assert call_args[1]["data"]["status"] == "running"
    assert call_args[1]["data"]["message"] == "Training in progress"


def test_send_error():
    """Test error message sending."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_7")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    sender.send_error(error_message="Out of memory", error_type="training_error")

    call_args = mock_manager.publish_event.call_args[0]
    assert call_args[1]["type"] == "error"
    assert call_args[1]["data"]["error"] == "Out of memory"
    assert call_args[1]["data"]["error_type"] == "training_error"


def test_send_early_stopping_triggered():
    """Test early stopping notification."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_8")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    sender.send_early_stopping_triggered(
        epoch=8,
        best_metric_value=0.933,
        metric_name="val_recall",
        patience=7
    )

    call_args = mock_manager.publish_event.call_args[0]
    payload = call_args[1]["data"]

    assert call_args[1]["type"] == "early_stopping"
    assert payload["epoch"] == 8
    assert payload["best_metric_value"] == 0.933
    assert payload["metric_name"] == "val_recall"
    assert payload["patience"] == 7


def test_send_training_mode():
    """Test training mode configuration signal."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_9")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    sender.send_training_mode(is_federated=True, num_rounds=5, num_clients=2)

    call_args = mock_manager.publish_event.call_args[0]
    payload = call_args[1]["data"]

    assert call_args[1]["type"] == "training_mode"
    assert payload["is_federated"] is True
    assert payload["num_rounds"] == 5
    assert payload["num_clients"] == 2


def test_send_round_end():
    """Test federated round end notification."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_10")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    fit_metrics = {"loss": 0.4}
    eval_metrics = {"accuracy": 0.88, "recall": 0.91}

    sender.send_round_end(
        round_num=3,
        total_rounds=5,
        fit_metrics=fit_metrics,
        eval_metrics=eval_metrics
    )

    call_args = mock_manager.publish_event.call_args[0]
    payload = call_args[1]["data"]

    assert call_args[1]["type"] == "round_end"
    assert payload["round"] == 3
    assert payload["total_rounds"] == 5
    assert payload["fit_metrics"] == fit_metrics
    assert payload["eval_metrics"] == eval_metrics


def test_send_round_metrics():
    """Test round metrics aggregation."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_11")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    metrics = {"loss": 0.35, "accuracy": 0.89}
    sender.send_round_metrics(round_num=2, total_rounds=5, metrics=metrics)

    call_args = mock_manager.publish_event.call_args[0]
    payload = call_args[1]["data"]

    assert call_args[1]["type"] == "round_metrics"
    assert payload["round"] == 2
    assert payload["total_rounds"] == 5
    assert payload["metrics"] == metrics


def test_send_gradient_stats():
    """Test gradient statistics sending."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_12")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    layer_norms = {"model.encoder": 0.5, "model.decoder": 0.3}
    sender.send_gradient_stats(
        step=100,
        total_norm=0.8,
        layer_norms=layer_norms,
        max_norm=0.5,
        min_norm=0.3
    )

    call_args = mock_manager.publish_event.call_args[0]
    payload = call_args[1]["data"]

    assert call_args[1]["type"] == "gradient_stats"
    assert payload["step"] == 100
    assert payload["total_norm"] == 0.8
    assert payload["layer_norms"] == layer_norms
    assert payload["max_norm"] == 0.5
    assert payload["min_norm"] == 0.3


def test_send_lr_update():
    """Test learning rate update notification."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_13")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    sender.send_lr_update(
        current_lr=0.001,
        step=500,
        epoch=5,
        scheduler_type="StepLR"
    )

    call_args = mock_manager.publish_event.call_args[0]
    payload = call_args[1]["data"]

    assert call_args[1]["type"] == "lr_update"
    assert payload["current_lr"] == 0.001
    assert payload["step"] == 500
    assert payload["epoch"] == 5
    assert payload["scheduler_type"] == "StepLR"


def test_send_training_end():
    """Test training end notification with run_id."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_14")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    summary_data = {
        "total_epochs": 10,
        "best_recall": 0.933,
        "final_loss": 0.245
    }

    sender.send_training_end(run_id=42, summary_data=summary_data)

    call_args = mock_manager.publish_event.call_args[0]
    payload = call_args[1]["data"]

    assert call_args[1]["type"] == "training_end"
    assert payload["run_id"] == 42
    assert payload["total_epochs"] == 10
    assert payload["best_recall"] == 0.933
    assert payload["final_loss"] == 0.245


def test_error_handling_in_send_metrics():
    """Test that errors in send_metrics are caught and logged."""
    sender = MetricsSSESender(experiment_id="test_exp_sender_15")

    # Mock event manager that raises exception
    mock_manager = MagicMock()
    mock_manager.publish_event.side_effect = Exception("Test error")
    sender._event_manager = mock_manager

    # Should not raise exception
    sender.send_metrics({"loss": 0.5}, "test")

    # Verify it attempted to publish
    mock_manager.publish_event.assert_called_once()


def test_sender_uses_global_event_manager():
    """Test that sender uses the global singleton event manager."""
    sender = MetricsSSESender(experiment_id="test_global_manager")

    # The sender should have a reference to the event manager
    assert sender._event_manager is not None

    # Create another sender - should share same event manager
    sender2 = MetricsSSESender(experiment_id="test_global_manager_2")
    assert sender._event_manager is sender2._event_manager


def test_close_is_noop():
    """Test that close() is a no-op for compatibility."""
    sender = MetricsSSESender(experiment_id="test_close")
    # Should not raise any exception
    sender.close()


def test_del_is_noop():
    """Test that __del__() is a no-op for compatibility."""
    sender = MetricsSSESender(experiment_id="test_del")
    # Should not raise any exception
    del sender


def test_publish_returns_success():
    """Test that successful publish is logged correctly."""
    sender = MetricsSSESender(experiment_id="test_success")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = True
    sender._event_manager = mock_manager

    # This should complete without warning
    sender.send_metrics({"value": 1}, "test")

    mock_manager.publish_event.assert_called_once()


def test_publish_returns_failure():
    """Test that failed publish is handled correctly."""
    sender = MetricsSSESender(experiment_id="test_failure")

    mock_manager = MagicMock()
    mock_manager.publish_event.return_value = False
    sender._event_manager = mock_manager

    # This should complete without raising, just log warning
    sender.send_metrics({"value": 1}, "test")

    mock_manager.publish_event.assert_called_once()
