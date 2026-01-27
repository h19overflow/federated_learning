"""
Shared step logic for training, validation, and test steps in LitResNetEnhanced.
"""

from typing import Tuple

import pytorch_lightning as pl
import torch

from federated_pneumonia_detection.src.control.dl_model.internals.model.utils.metrics_handler import (  # noqa: E501
    MetricsHandler,
)


class StepLogic:
    """
    Shared step logic for training, validation, and test steps.
    Reduces duplication by centralizing the forward, loss, and metric update flow.
    """

    def __init__(self, metrics_handler: MetricsHandler):
        """
        Initialize step logic.

        Args:
            metrics_handler: The metrics handler to update and log with
        """
        self.metrics_handler = metrics_handler

    def execute_step(
        self,
        pl_module: pl.LightningModule,
        batch: Tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        """
        Execute a shared step (forward, loss, metrics update, logging).

        Args:
            pl_module: The LightningModule instance
            batch: The batch of data (x, y)
            stage: The current stage ('train', 'val', or 'test')

        Returns:
            The calculated loss
        """
        x, y = batch
        logits = pl_module(x)
        loss = pl_module._calculate_loss(logits, y)

        preds = torch.sigmoid(logits)
        targets_for_metrics = y.int().unsqueeze(1) if y.dim() == 1 else y.int()

        self.metrics_handler.update(stage, preds, targets_for_metrics)
        self.metrics_handler.log(stage, pl_module, loss)

        return loss
