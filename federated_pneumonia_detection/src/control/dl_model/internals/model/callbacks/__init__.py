"""
Callbacks module for PyTorch Lightning training.

Provides callback classes and setup utilities for training orchestration,
including early stopping, checkpointing, batch metrics, and gradient monitoring.
"""

from federated_pneumonia_detection.src.control.dl_model.internals.model.callbacks.batch_metrics import (
    BatchMetricsCallback,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.callbacks.checkpoint import (
    HighestValRecallCallback,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.callbacks.early_stopping import (
    EarlyStoppingSignalCallback,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.callbacks.gradient_monitor import (
    GradientMonitorCallback,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.callbacks.setup import (
    compute_class_weights_for_pl,
    create_trainer_from_config,
    prepare_trainer_and_callbacks_pl,
)

__all__ = [
    # Callback classes
    "EarlyStoppingSignalCallback",
    "HighestValRecallCallback",
    "BatchMetricsCallback",
    "GradientMonitorCallback",
    # Setup functions
    "compute_class_weights_for_pl",
    "prepare_trainer_and_callbacks_pl",
    "create_trainer_from_config",
]
