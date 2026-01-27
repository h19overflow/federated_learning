"""
Metric initialization and management for LitResNetEnhanced.
"""

from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class MetricsHandler(nn.Module):
    """
    Handles initialization, update, and logging of metrics for LitResNetEnhanced.
    Provides a clean interface to manage 13 different metrics across train/val/test stages.  # noqa: E501
    """

    def __init__(self, num_classes: int):
        """
        Initialize metrics for all stages.

        Args:
            num_classes: Number of output classes (1 for binary)
        """
        super().__init__()
        self.num_classes = num_classes
        task_type = "binary" if num_classes == 1 else "multiclass"
        actual_num_classes = 2 if num_classes == 1 else num_classes

        # Training metrics
        self.train_accuracy = torchmetrics.Accuracy(
            task=task_type,
            num_classes=actual_num_classes,
        )
        self.train_f1 = torchmetrics.F1Score(
            task=task_type,
            num_classes=actual_num_classes,
        )

        # Validation metrics
        self.val_accuracy = torchmetrics.Accuracy(
            task=task_type,
            num_classes=actual_num_classes,
        )
        self.val_precision = torchmetrics.Precision(
            task=task_type,
            num_classes=actual_num_classes,
        )
        self.val_recall = torchmetrics.Recall(
            task=task_type,
            num_classes=actual_num_classes,
        )
        self.val_f1 = torchmetrics.F1Score(
            task=task_type,
            num_classes=actual_num_classes,
        )
        self.val_auroc = torchmetrics.AUROC(
            task=task_type,
            num_classes=actual_num_classes,
        )
        self.val_confusion = torchmetrics.ConfusionMatrix(
            task=task_type,
            num_classes=actual_num_classes,
        )

        # Test metrics
        self.test_accuracy = torchmetrics.Accuracy(
            task=task_type,
            num_classes=actual_num_classes,
        )
        self.test_precision = torchmetrics.Precision(
            task=task_type,
            num_classes=actual_num_classes,
        )
        self.test_recall = torchmetrics.Recall(
            task=task_type,
            num_classes=actual_num_classes,
        )
        self.test_f1 = torchmetrics.F1Score(
            task=task_type,
            num_classes=actual_num_classes,
        )
        self.test_auroc = torchmetrics.AUROC(
            task=task_type,
            num_classes=actual_num_classes,
        )

    def update(self, stage: str, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics for the given stage.

        Args:
            stage: Current stage ('train', 'val', or 'test')
            preds: Model predictions
            targets: Ground truth targets
        """
        if stage == "train":
            self.train_accuracy.update(preds, targets)
            self.train_f1.update(preds, targets)
        elif stage == "val":
            self.val_accuracy.update(preds, targets)
            self.val_precision.update(preds, targets)
            self.val_recall.update(preds, targets)
            self.val_f1.update(preds, targets)
            self.val_auroc.update(preds, targets)
            self.val_confusion.update(preds, targets)
        elif stage == "test":
            self.test_accuracy.update(preds, targets)
            self.test_precision.update(preds, targets)
            self.test_recall.update(preds, targets)
            self.test_f1.update(preds, targets)
            self.test_auroc.update(preds, targets)

    def log(
        self,
        stage: str,
        pl_module: pl.LightningModule,
        loss: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Log metrics for the given stage.

        Args:
            stage: Current stage ('train', 'val', or 'test')
            pl_module: The LightningModule instance to log to
            loss: Optional loss value to log
        """
        if stage == "train":
            if loss is not None:
                pl_module.log(
                    "train_loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            pl_module.log(
                "train_acc",
                self.train_accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            pl_module.log(
                "train_f1",
                self.train_f1,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        elif stage == "val":
            if loss is not None:
                pl_module.log(
                    "val_loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
            pl_module.log(
                "val_acc",
                self.val_accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            pl_module.log(
                "val_precision",
                self.val_precision,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            pl_module.log(
                "val_recall",
                self.val_recall,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            pl_module.log(
                "val_f1",
                self.val_f1,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            pl_module.log(
                "val_auroc",
                self.val_auroc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        elif stage == "test":
            if loss is not None:
                pl_module.log(
                    "test_loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            pl_module.log(
                "test_acc",
                self.test_accuracy,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            pl_module.log(
                "test_precision",
                self.test_precision,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            pl_module.log(
                "test_recall",
                self.test_recall,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            pl_module.log(
                "test_f1",
                self.test_f1,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            pl_module.log(
                "test_auroc",
                self.test_auroc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def get_confusion_matrix_metrics(self) -> Dict[str, int]:
        """
        Compute and return confusion matrix metrics.

        Returns:
            Dictionary containing TN, FP, FN, TP
        """
        cm = self.val_confusion.compute()
        return {
            "val_cm_tn": int(cm[0, 0].item()),
            "val_cm_fp": int(cm[0, 1].item()),
            "val_cm_fn": int(cm[1, 0].item()),
            "val_cm_tp": int(cm[1, 1].item()),
        }

    def reset_confusion_matrix(self) -> None:
        """Reset the validation confusion matrix."""
        self.val_confusion.reset()
