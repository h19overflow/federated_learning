"""
Enhanced PyTorch Lightning module for ResNet-based pneumonia classification.
Includes Focal Loss, Cosine Annealing with Warmup, and progressive unfreezing.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchvision.models import ResNet50_Weights

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

from federated_pneumonia_detection.src.control.dl_model.internals.model.losses import (
    FocalLoss,
    FocalLossWithLabelSmoothing,
)
from federated_pneumonia_detection.src.entities.resnet_with_custom_head import (
    ResNetWithCustomHead,
)


class LitResNetEnhanced(pl.LightningModule):
    """
    Enhanced PyTorch Lightning module for high-accuracy pneumonia classification.

    Includes:
    - Focal Loss for class imbalance
    - Cosine Annealing with Warmup
    - Progressive unfreezing support
    - Label smoothing option
    """

    def __init__(
        self,
        config: Optional["ConfigManager"] = None,
        base_model_weights: Optional[ResNet50_Weights] = None,
        class_weights_tensor: Optional[torch.Tensor] = None,
        num_classes: int = 1,
        monitor_metric: str = "val_recall",
        # Enhanced options
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        use_cosine_scheduler: bool = True,
        warmup_epochs: int = 3,
    ):
        """
        Initialize enhanced Lightning module.

        Args:
            config: ConfigManager for configuration
            base_model_weights: Optional ResNet50 weights
            class_weights_tensor: Optional class weights for loss
            num_classes: Number of output classes (1 for binary)
            monitor_metric: Metric to monitor for scheduling
            use_focal_loss: Whether to use Focal Loss
            focal_alpha: Alpha parameter for Focal Loss
            focal_gamma: Gamma parameter for Focal Loss
            label_smoothing: Label smoothing factor (0 to disable)
            use_cosine_scheduler: Use cosine annealing with warmup
            warmup_epochs: Number of warmup epochs
        """
        super().__init__()

        if config is None:
            from federated_pneumonia_detection.config.config_manager import (
                ConfigManager,
            )

            config = ConfigManager()

        self.config = config
        self.num_classes = num_classes
        self.monitor_metric = monitor_metric
        self.logger_obj = logging.getLogger(__name__)

        # Enhanced options
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.use_cosine_scheduler = use_cosine_scheduler
        self.warmup_epochs = warmup_epochs

        # Save hyperparameters
        self.save_hyperparameters(
            ignore=["config", "base_model_weights", "class_weights_tensor"],
        )

        # Validate configuration
        self._validate_config()

        # Initialize model
        self.model = ResNetWithCustomHead(
            config=self.config,
            base_model_weights=base_model_weights,
            num_classes=num_classes,
            dropout_rate=self.config.get("experiment.dropout_rate", 0.5),
            fine_tune_layers_count=self.config.get(
                "experiment.fine_tune_layers_count",
                0,
            ),
        )

        # Apply torch.compile if enabled (PyTorch 2.0+ performance optimization)
        if self.config.get("experiment.use_torch_compile", False):
            compile_mode = self.config.get("experiment.torch_compile_mode", "default")
            self.logger_obj.info(f"Applying torch.compile with mode='{compile_mode}'")
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
                self.logger_obj.info("Model compiled successfully")
            except Exception as e:
                self.logger_obj.warning(
                    f"torch.compile failed, falling back to eager mode: {e}",
                )

        # Store class weights
        self.class_weights_tensor = class_weights_tensor

        # Initialize metrics
        self._setup_metrics()

        # Setup loss function
        self._setup_loss_function()

        # Progressive unfreezing state
        self.unfrozen_layers = 0

        self.logger_obj.info(
            f"LitResNetEnhanced initialized with {self.model.get_model_info()['total_parameters']} parameters",
        )
        self.logger_obj.info(
            f"Loss: {'Focal' if use_focal_loss else 'BCE'}, "
            f"Scheduler: {'CosineWarmup' if use_cosine_scheduler else 'ReduceLROnPlateau'}",
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        lr = self.config.get("experiment.learning_rate", 0)
        if lr <= 0:
            raise ValueError("Learning rate must be positive")

        wd = self.config.get("experiment.weight_decay", -1)
        if wd < 0:
            raise ValueError("Weight decay must be non-negative")

    def _setup_metrics(self) -> None:
        """Initialize torchmetrics for tracking performance."""
        num_classes = 2 if self.num_classes == 1 else self.num_classes
        task_type = "binary" if self.num_classes == 1 else "multiclass"

        # Training metrics
        self.train_accuracy = torchmetrics.Accuracy(
            task=task_type,
            num_classes=num_classes,
        )
        self.train_f1 = torchmetrics.F1Score(task=task_type, num_classes=num_classes)

        # Validation metrics
        self.val_accuracy = torchmetrics.Accuracy(
            task=task_type,
            num_classes=num_classes,
        )
        self.val_precision = torchmetrics.Precision(
            task=task_type,
            num_classes=num_classes,
        )
        self.val_recall = torchmetrics.Recall(task=task_type, num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task=task_type, num_classes=num_classes)
        self.val_auroc = torchmetrics.AUROC(task=task_type, num_classes=num_classes)
        self.val_confusion = torchmetrics.ConfusionMatrix(
            task=task_type,
            num_classes=num_classes,
        )

        # Test metrics
        self.test_accuracy = torchmetrics.Accuracy(
            task=task_type,
            num_classes=num_classes,
        )
        self.test_precision = torchmetrics.Precision(
            task=task_type,
            num_classes=num_classes,
        )
        self.test_recall = torchmetrics.Recall(task=task_type, num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task=task_type, num_classes=num_classes)
        self.test_auroc = torchmetrics.AUROC(task=task_type, num_classes=num_classes)

    def _setup_loss_function(self) -> None:
        """Setup loss function with enhanced options."""
        pos_weight = None
        if self.class_weights_tensor is not None:
            pos_weight = self.class_weights_tensor[1] / (
                self.class_weights_tensor[0] + 1e-8
            )
            self.logger_obj.info(f"Using positive class weight: {pos_weight}")

        if self.use_focal_loss:
            if self.label_smoothing > 0:
                self.loss_fn = FocalLossWithLabelSmoothing(
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    smoothing=self.label_smoothing,
                    pos_weight=pos_weight,
                )
                self.logger_obj.info(
                    f"Using FocalLoss with label smoothing ({self.label_smoothing})",
                )
            else:
                self.loss_fn = FocalLoss(
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    pos_weight=pos_weight,
                )
                self.logger_obj.info("Using FocalLoss")
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.logger_obj.info("Using BCEWithLogitsLoss")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _calculate_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate loss based on task type."""
        targets = (
            targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()
        )
        return self.loss_fn(logits, targets)

    def _get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to predictions."""
        return torch.sigmoid(logits)

    def _prepare_targets_for_metrics(self, targets: torch.Tensor) -> torch.Tensor:
        """Prepare targets for metric computation."""
        return targets.int().unsqueeze(1) if targets.dim() == 1 else targets.int()

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform training step."""
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)

        preds = self._get_predictions(logits)
        targets_for_metrics = self._prepare_targets_for_metrics(y)

        self.train_accuracy.update(preds, targets_for_metrics)
        self.train_f1.update(preds, targets_for_metrics)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_f1",
            self.train_f1,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform validation step."""
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)

        preds = self._get_predictions(logits)
        targets_for_metrics = self._prepare_targets_for_metrics(y)

        self.val_accuracy.update(preds, targets_for_metrics)
        self.val_precision.update(preds, targets_for_metrics)
        self.val_recall.update(preds, targets_for_metrics)
        self.val_f1.update(preds, targets_for_metrics)
        self.val_auroc.update(preds, targets_for_metrics)
        self.val_confusion.update(preds, targets_for_metrics)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_acc",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "val_auroc",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform test step."""
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)

        preds = self._get_predictions(logits)
        targets_for_metrics = self._prepare_targets_for_metrics(y)

        self.test_accuracy.update(preds, targets_for_metrics)
        self.test_precision.update(preds, targets_for_metrics)
        self.test_recall.update(preds, targets_for_metrics)
        self.test_f1.update(preds, targets_for_metrics)
        self.test_auroc.update(preds, targets_for_metrics)

        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_acc",
            self.test_accuracy,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_precision",
            self.test_precision,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_recall",
            self.test_recall,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_auroc",
            self.test_auroc,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.get("experiment.learning_rate", 0.001),
            weight_decay=self.config.get("experiment.weight_decay", 0.0001),
        )

        if self.use_cosine_scheduler:
            total_epochs = self.config.get("experiment.epochs", 10)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(total_epochs // 3, 1),
                T_mult=1,
                eta_min=self.config.get("experiment.min_lr", 1e-7),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max" if "recall" in self.monitor_metric else "min",
                factor=self.config.get("experiment.reduce_lr_factor", 0.5),
                patience=self.config.get("experiment.reduce_lr_patience", 3),
                min_lr=self.config.get("experiment.min_lr", 1e-7),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                },
            }

    def on_train_epoch_start(self) -> None:
        """Called at the start of each training epoch."""
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        cm = self.val_confusion.compute()
        tn = int(cm[0, 0].item())
        fp = int(cm[0, 1].item())
        fn = int(cm[1, 0].item())
        tp = int(cm[1, 1].item())

        self.log("val_cm_tn", tn, on_epoch=True, sync_dist=True)
        self.log("val_cm_fp", fp, on_epoch=True, sync_dist=True)
        self.log("val_cm_fn", fn, on_epoch=True, sync_dist=True)
        self.log("val_cm_tp", tp, on_epoch=True, sync_dist=True)

        self.val_confusion.reset()

    def progressive_unfreeze(self, layers_to_unfreeze: int = 1) -> None:
        """
        Progressively unfreeze backbone layers.

        Args:
            layers_to_unfreeze: Number of layers to unfreeze from the end
        """
        self.unfrozen_layers += layers_to_unfreeze
        self.model._unfreeze_last_n_layers(self.unfrozen_layers)
        self.logger_obj.info(
            f"Progressive unfreeze: {self.unfrozen_layers} layers unfrozen",
        )

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        self.model.freeze_backbone()
        self.unfrozen_layers = 0

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        self.model.unfreeze_backbone()

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        model_info = self.model.get_model_info()
        model_info.update(
            {
                "lightning_module": "LitResNetEnhanced",
                "optimizer": "AdamW",
                "learning_rate": self.config.get("experiment.learning_rate", 0),
                "weight_decay": self.config.get("experiment.weight_decay", 0),
                "scheduler": "CosineAnnealingWarmRestarts"
                if self.use_cosine_scheduler
                else "ReduceLROnPlateau",
                "loss_function": "FocalLoss"
                if self.use_focal_loss
                else "BCEWithLogitsLoss",
                "label_smoothing": self.label_smoothing,
                "focal_alpha": self.focal_alpha if self.use_focal_loss else None,
                "focal_gamma": self.focal_gamma if self.use_focal_loss else None,
                "unfrozen_layers": self.unfrozen_layers,
                "device": str(self.device),
            },
        )
        return model_info
