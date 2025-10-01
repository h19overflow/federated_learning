"""
Training callbacks and utilities for PyTorch Lightning model training.
Provides checkpoint management, early stopping, and monitoring functionality.
"""

import os
import logging
from typing import Optional, List, Dict, Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
import numpy as np
from sklearn.utils import class_weight

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig


class HighestValRecallCallback(pl.Callback):
    """Custom callback to track highest validation recall achieved during training."""

    def __init__(self):
        super().__init__()
        self.best_recall = 0.0
        self.logger = logging.getLogger(__name__)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Track the highest validation recall."""
        current_recall = trainer.callback_metrics.get('val_recall', 0.0)
        if isinstance(current_recall, torch.Tensor):
            current_recall = current_recall.item()

        if current_recall > self.best_recall:
            self.best_recall = current_recall
            self.logger.info(f"New best validation recall: {self.best_recall:.4f}")


def compute_class_weights_for_pl(train_df, class_column: str = 'Target') -> Optional[torch.Tensor]:
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
                'balanced',
                classes=unique_labels,
                y=labels
            )
            return torch.tensor(weights, dtype=torch.float)
        else:
            logging.getLogger(__name__).warning(f"Expected 2 classes, found {len(unique_labels)}")
            return None

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to compute class weights: {e}")
        return None


def prepare_trainer_and_callbacks_pl(
    train_df_for_weights,
    class_column: str = 'Target',
    checkpoint_dir: str = 'checkpoints_pl',
    model_filename: str = 'best_model',
    constants: Optional[SystemConstants] = None,
    config: Optional[ExperimentConfig] = None
) -> Dict[str, Any]:
    """
    Prepare PyTorch Lightning trainer callbacks and configuration.

    Args:
        train_df_for_weights: Training dataframe for computing class weights
        class_column: Column name containing class labels
        checkpoint_dir: Directory to save model checkpoints
        model_filename: Base filename for saved models
        constants: System constants for configuration
        config: Experiment configuration

    Returns:
        Dictionary containing callbacks and trainer configuration
    """
    logger = logging.getLogger(__name__)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup default values from config or fallbacks
    patience = config.early_stopping_patience if config else 7
    min_delta = getattr(config, 'early_stopping_min_delta', 0.001)
    reduce_lr_patience = config.reduce_lr_patience if config else 3
    max_epochs = config.epochs if config else 50

    # Compute class weights
    class_weights = compute_class_weights_for_pl(train_df_for_weights, class_column)

    # ModelCheckpoint callback - save best model based on validation recall
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{model_filename}_{{epoch:02d}}_{{val_recall:.3f}}',
        monitor='val_recall',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True
    )

    # Additional checkpoint for validation loss (backup metric)
    checkpoint_loss_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{model_filename}_loss_{{epoch:02d}}_{{val_loss:.3f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        auto_insert_metric_name=False,
        verbose=False
    )

    # EarlyStopping callback - stop training when validation recall stops improving
    early_stop_callback = EarlyStopping(
        monitor='val_recall',
        mode='max',
        patience=patience,
        min_delta=min_delta,
        verbose=True,
        strict=True
    )

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(
        logging_interval='epoch',
        log_momentum=True
    )

    # Custom highest recall tracker
    highest_recall_callback = HighestValRecallCallback()

    # Compile callbacks list
    callbacks = [
        checkpoint_callback,
        checkpoint_loss_callback,
        early_stop_callback,
        lr_monitor,
        highest_recall_callback
    ]

    # Trainer configuration
    trainer_config = {
        'callbacks': callbacks,
        'max_epochs': max_epochs,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1 if torch.cuda.is_available() else 'auto',
        'precision': '16-mixed' if torch.cuda.is_available() else 32,
        'log_every_n_steps': 50,
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'deterministic': True if constants and constants.random_seed is not None else False
    }

    logger.info(f"Prepared trainer with {len(callbacks)} callbacks")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Early stopping patience: {patience}")
    logger.info(f"Class weights computed: {class_weights is not None}")

    return {
        'callbacks': callbacks,
        'trainer_config': trainer_config,
        'class_weights': class_weights,
        'checkpoint_dir': checkpoint_dir
    }


def create_trainer_from_config(
    constants: SystemConstants,
    config: ExperimentConfig,
    callbacks: List[pl.Callback]
) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer with proper configuration.

    Args:
        constants: System constants
        config: Experiment configuration
        callbacks: List of callbacks to use

    Returns:
        Configured PyTorch Lightning trainer
    """
    # Set deterministic training if seed is provided
    if constants.SEED is not None:
        pl.seed_everything(constants.SEED, workers=True)

    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=config.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else 'auto',
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=constants.SEED is not None,
        gradient_clip_val=getattr(config, 'gradient_clip_val', 1.0),
        accumulate_grad_batches=getattr(config, 'accumulate_grad_batches', 1)
    )

    logging.getLogger(__name__).info(f"Trainer created with {len(callbacks)} callbacks")
    return trainer