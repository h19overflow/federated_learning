"""
Setup functions for PyTorch Lightning trainer configuration and callbacks.
"""

import os
import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import torch
import numpy as np
from sklearn.utils import class_weight

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

from federated_pneumonia_detection.src.control.dl_model.utils.model.metrics_collector import (
    MetricsCollectorCallback,
)
from federated_pneumonia_detection.src.control.dl_model.utils.data.websocket_metrics_sender import (
    MetricsWebSocketSender,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.callbacks.early_stopping import (
    EarlyStoppingSignalCallback,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.callbacks.checkpoint import (
    HighestValRecallCallback,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.callbacks.batch_metrics import (
    BatchMetricsCallback,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.callbacks.gradient_monitor import (
    GradientMonitorCallback,
)


def compute_class_weights_for_pl(
    train_df, class_column: str = "Target"
) -> Optional[torch.Tensor]:
    """
    Compute balanced class weights for PyTorch Lightning training.

    Args:
        train_df: Training dataframe containing labels
        class_column: Column name containing class labels

    Returns:
        Tensor of class weights for binary classification, None if computation fails
    """
    try:
        labels = train_df[class_column].values
        unique_labels = np.unique(labels)

        if len(unique_labels) == 2:
            weights = class_weight.compute_class_weight(
                "balanced", classes=unique_labels, y=labels
            )
            return torch.tensor(weights, dtype=torch.float)
        else:
            logging.getLogger(__name__).warning(
                f"Expected 2 classes, found {len(unique_labels)}"
            )
            return None

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to compute class weights: {e}")
        return None


def prepare_trainer_and_callbacks_pl(
    train_df_for_weights,
    class_column: str = "Target",
    checkpoint_dir: str = "checkpoints_pl",
    model_filename: str = "best_model",
    config: Optional["ConfigManager"] = None,
    metrics_dir: Optional[str] = None,
    experiment_name: str = "pneumonia_detection",
    run_id: Optional[int] = None,
    enable_db_persistence: bool = True,
    websocket_sender: Optional[MetricsWebSocketSender] = None,
    is_federated: bool = False,
    client_id: Optional[int] = None,
    round_number: int = 0,
    batch_sample_interval: Optional[int] = None,
    gradient_sample_interval: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare PyTorch Lightning trainer callbacks and configuration.

    Args:
        train_df_for_weights: Training dataframe for computing class weights
        class_column: Column name containing class labels
        checkpoint_dir: Directory to save model checkpoints
        model_filename: Base filename for saved models
        config: ConfigManager instance
        metrics_dir: Optional directory to save metrics (defaults to checkpoint_dir/metrics)
        experiment_name: Name of the experiment for metrics tracking
        run_id: Optional database run ID for metrics persistence
        enable_db_persistence: Whether to persist metrics to database
        websocket_sender: Optional MetricsWebSocketSender instance for frontend communication
        is_federated: If True, uses local_epochs; if False, uses epochs
        client_id: Optional client ID for federated learning context
        round_number: Round number for federated learning
        batch_sample_interval: Send batch metrics every N batches (default: 10)
        gradient_sample_interval: Send gradient stats every N steps (default: 20)

    Returns:
        Dictionary containing callbacks and trainer configuration
    """
    if config is None:
        from federated_pneumonia_detection.config.config_manager import ConfigManager

        config = ConfigManager()

    logger = logging.getLogger(__name__)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create default WebSocket sender if not provided
    if websocket_sender is None:
        websocket_sender = MetricsWebSocketSender(websocket_uri="ws://localhost:8765")
        logger.info("[Training Callbacks] Created default WebSocket sender")

    # Setup default values from config
    patience = config.get("experiment.early_stopping_patience", 7)
    min_delta = config.get("experiment.early_stopping_min_delta", 0.001)
    max_epochs = (
        config.get("experiment.max-epochs", 50)
        if is_federated
        else config.get("experiment.epochs", 50)
    )
    training_mode = "federated" if is_federated else "centralized"

    logger.info(
        f"[Trainer Setup] max_epochs={max_epochs}, early_stopping_patience={patience}, min_delta={min_delta}"
    )
    if is_federated and client_id is not None:
        logger.info(
            f"[Trainer Setup] Federated mode with client_id={client_id}, round={round_number}"
        )

    # Compute class weights
    class_weights = compute_class_weights_for_pl(train_df_for_weights, class_column)

    # ModelCheckpoint callback - save best model based on validation recall
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{model_filename}_{{epoch:02d}}_{{val_recall:.3f}}",
        monitor="val_recall",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True,
    )
    # EarlyStopping callback - stop training when validation recall stops improving
    early_stop_callback = EarlyStopping(
        monitor="val_recall",
        mode="max",
        patience=patience,
        min_delta=min_delta,
        verbose=True,
        strict=True,
        log_rank_zero_only=True,
    )

    logger.info(
        f"[EarlyStopping] Monitoring 'val_recall' with patience={patience}, min_delta={min_delta}"
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(
        logging_interval="epoch",
        log_momentum=True,
    )

    # Custom highest recall tracker
    highest_recall_callback = HighestValRecallCallback()

    # Metrics collector - save all training metrics
    if metrics_dir is None:
        metrics_dir = os.path.join(checkpoint_dir, "metrics")
    metrics_collector = MetricsCollectorCallback(
        save_dir=metrics_dir,
        experiment_name=experiment_name,
        run_id=run_id,
        training_mode=training_mode,
        enable_db_persistence=True,
        websocket_uri="ws://localhost:8765",
        client_id=client_id,
        round_number=round_number,
    )

    # Early stopping signal callback - notify frontend when early stopping occurs
    early_stopping_signal = EarlyStoppingSignalCallback(
        websocket_sender=websocket_sender
    )

    # Batch metrics callback - send batch-level metrics for real-time observability
    batch_interval = batch_sample_interval or config.get("experiment.batch_sample_interval", 10)
    batch_metrics_callback = BatchMetricsCallback(
        websocket_sender=websocket_sender,
        sample_interval=batch_interval,
        client_id=client_id,
        round_num=round_number,
    )

    # Gradient monitor callback - track gradient norms and learning rate
    gradient_interval = gradient_sample_interval or config.get("experiment.gradient_sample_interval", 20)
    gradient_monitor_callback = GradientMonitorCallback(
        websocket_sender=websocket_sender,
        sample_interval=gradient_interval,
    )

    # Compile callbacks list
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        early_stopping_signal,  # Must come after early_stop_callback
        lr_monitor,
        highest_recall_callback,
        metrics_collector,
        batch_metrics_callback,
        gradient_monitor_callback,
    ]

    # Trainer configuration
    trainer_config = {
        "callbacks": callbacks,
        "max_epochs": max_epochs,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1 if torch.cuda.is_available() else "auto",
        "precision": "16-mixed" if torch.cuda.is_available() else 32,
        "log_every_n_steps": 50,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "deterministic": True if config.get("system.seed") is not None else False,
    }

    logger.info(f"Prepared trainer with {len(callbacks)} callbacks")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Early stopping patience: {patience}")
    logger.info(f"Class weights computed: {class_weights is not None}")

    return {
        "callbacks": callbacks,
        "trainer_config": trainer_config,
        "class_weights": class_weights,
        "checkpoint_dir": checkpoint_dir,
        "metrics_collector": metrics_collector,
        "early_stopping_signal": early_stopping_signal,
    }


def create_trainer_from_config(
    config: Optional["ConfigManager"],
    callbacks: List[pl.Callback],
    is_federated: bool = False,
) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer with proper configuration.

    Args:
        config: ConfigManager instance
        callbacks: List of callbacks to use
        is_federated: If True, uses local_epochs; if False, uses epochs

    Returns:
        Configured PyTorch Lightning trainer
    """
    if config is None:
        from federated_pneumonia_detection.config.config_manager import ConfigManager

        config = ConfigManager()

    # Set deterministic training if seed is provided
    seed = config.get("system.seed")
    if seed is not None:
        pl.seed_everything(seed, workers=True)

    epochs = (
        config.get("experiment.max-epochs", 50)
        if is_federated
        else config.get("experiment.epochs", 50)
    )
    gradient_clip_val = config.get("experiment.gradient_clip_val", 1.0)
    accumulate_grad_batches = config.get("experiment.accumulate_grad_batches", 1)

    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else "auto",
        precision=32,  # Force FP32 to avoid focal loss underflow (FP16 causes gradient underflow)
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=seed is not None,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    logging.getLogger(__name__).info(f"Trainer created with {len(callbacks)} callbacks")
    return trainer
