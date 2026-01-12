"""Model and trainer setup utilities."""

import os
import logging
from typing import Optional, Tuple, Any

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.utils.model.callbacks.setup import (
    prepare_trainer_and_callbacks_pl,
    create_trainer_from_config,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet_enhanced import (
    LitResNetEnhanced,
    ProgressiveUnfreezeCallback,
)


def build_model_and_callbacks(
    train_df: pd.DataFrame,
    config: ConfigManager,
    checkpoint_dir: str,
    logs_dir: str,
    logger: logging.Logger,
    experiment_name: str = "pneumonia_detection",
    run_id: Optional[int] = None,
    is_federated: bool = False,
    client_id: Optional[int] = None,
    round_number: int = 0,
) -> Tuple[pl.LightningModule, list, Any]:
    """
    Build model and training callbacks.

    Args:
        train_df: Training dataframe for computing class weights
        config: Configuration manager instance
        checkpoint_dir: Directory for model checkpoints
        logs_dir: Directory for training logs
        logger: Logger instance
        experiment_name: Name for the experiment
        run_id: Optional database run ID for metrics persistence
        is_federated: If True, uses local_epochs; if False, uses epochs
        client_id: Optional client ID for federated learning context
        round_number: Round number for federated learning

    Returns:
        Tuple of (model, callbacks, metrics_collector)
    """
    logger.info("Setting up model and callbacks (Enhanced v3 - Balanced)...")
    if is_federated and client_id is not None:
        logger.info(
            f"[CentralizedTrainer] Federated mode enabled for client_id={client_id}, round={round_number}"
        )

    batch_interval = config.get("experiment.batch_sample_interval", 10)
    gradient_interval = config.get("experiment.gradient_sample_interval", 20)

    callback_config = prepare_trainer_and_callbacks_pl(
        train_df_for_weights=train_df,
        class_column=config.get("columns.target"),
        checkpoint_dir=checkpoint_dir,
        model_filename="pneumonia_model",
        config=config,
        metrics_dir=os.path.join(logs_dir, "metrics"),
        experiment_name=experiment_name,
        run_id=run_id,
        enable_db_persistence=True,
        is_federated=is_federated,
        client_id=client_id,
        round_number=round_number,
        batch_sample_interval=batch_interval,
        gradient_sample_interval=gradient_interval,
    )

    model = LitResNetEnhanced(
        config=config,
        class_weights_tensor=callback_config["class_weights"],
        use_focal_loss=True,
        focal_alpha=0.6,
        focal_gamma=1.5,
        label_smoothing=0.05,
        use_cosine_scheduler=True,
        monitor_metric="val_recall",
    )

    total_epochs = config.get("experiment.epochs", 25)
    unfreeze_epochs = [
        int(total_epochs * 0.15),
        int(total_epochs * 0.35),
        int(total_epochs * 0.55),
        int(total_epochs * 0.75),
    ]
    progressive_callback = ProgressiveUnfreezeCallback(
        unfreeze_epochs=unfreeze_epochs,
        layers_per_unfreeze=4,
    )
    logger.info(f"Progressive unfreezing at epochs: {unfreeze_epochs}")

    callbacks = callback_config["callbacks"] + [progressive_callback]

    return model, callbacks, callback_config["metrics_collector"]


def build_trainer(
    config: ConfigManager,
    callbacks: list,
    logs_dir: str,
    experiment_name: str,
    logger: logging.Logger,
    is_federated: bool = False,
) -> pl.Trainer:
    """
    Build PyTorch Lightning trainer.

    Args:
        config: Configuration manager instance
        callbacks: List of callbacks to use
        logs_dir: Directory for training logs
        experiment_name: Name for experiment logging
        logger: Logger instance
        is_federated: If True, uses max-epochs; if False, uses epochs

    Returns:
        Configured trainer instance
    """
    logger.info("Setting up trainer...")
    try:
        tb_logger = TensorBoardLogger(
            save_dir=logs_dir, name=experiment_name, version=None
        )
    except Exception as e:
        logger.error(f"Failed to create TensorBoard logger: {e}")
        tb_logger = None

    try:
        trainer = create_trainer_from_config(
            config=config, callbacks=callbacks, is_federated=is_federated
        )
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        trainer = None

    if tb_logger:
        trainer.logger = tb_logger
    return trainer
