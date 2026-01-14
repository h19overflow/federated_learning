"""
Enhanced PyTorch Lightning module for ResNet-based pneumonia classification.
Includes Focal Loss, Cosine Annealing with Warmup, and progressive unfreezing.
"""

import logging
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig, LRSchedulerTypeUnion, OptimizerConfig
from torchvision.models import ResNet50_Weights
from torchmetrics import Metric

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

from federated_pneumonia_detection.src.entities.resnet_with_custom_head import ResNetWithCustomHead
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_model_helpers import (
    validate_config,
    setup_metrics,
    setup_loss_function,
    calculate_loss,
    get_predictions,
    prepare_targets_for_metrics,
    get_model_summary,
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
            from federated_pneumonia_detection.config.config_manager import ConfigManager
            config = ConfigManager()

        self.config: "ConfigManager" = config
        self.num_classes: int = num_classes
        self.monitor_metric: str = monitor_metric
        self.logger_obj: logging.Logger = logging.getLogger(__name__)

        self.use_focal_loss: bool = use_focal_loss
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.label_smoothing: float = label_smoothing
        self.use_cosine_scheduler: bool = use_cosine_scheduler
        self.warmup_epochs: int = warmup_epochs

        self.save_hyperparameters(ignore=[
            "config", "base_model_weights", "class_weights_tensor"
        ])

        self._validate_config()

        self.model: ResNetWithCustomHead = ResNetWithCustomHead(
            config=self.config,
            base_model_weights=base_model_weights,
            num_classes=num_classes,
            dropout_rate=self.config.get("experiment.dropout_rate", 0.5),
            fine_tune_layers_count=self.config.get("experiment.fine_tune_layers_count", 0)
        )

        if self.config.get('experiment.use_torch_compile', False):
            compile_mode = self.config.get('experiment.torch_compile_mode', 'default')
            self.logger_obj.info(f"Applying torch.compile with mode='{compile_mode}'")
            try:
                self.model = torch.compile(self.model, mode=compile_mode)  # type: ignore[assignment]
                self.logger_obj.info("Model compiled successfully")
            except Exception as e:
                self.logger_obj.warning(f"torch.compile failed, falling back to eager mode: {e}")

        self.class_weights_tensor: Optional[torch.Tensor] = class_weights_tensor

        self.train_accuracy: Metric
        self.train_f1: Metric
        self.val_accuracy: Metric
        self.val_precision: Metric
        self.val_recall: Metric
        self.val_f1: Metric
        self.val_auroc: Metric
        self.val_confusion: Metric
        self.test_accuracy: Metric
        self.test_precision: Metric
        self.test_recall: Metric
        self.test_f1: Metric
        self.test_auroc: Metric

        self._setup_metrics()
        self._setup_loss_function()
        self.unfrozen_layers: int = 0

        self.logger_obj.info(
            f"LitResNetEnhanced initialized with {self.model.get_model_info()['total_parameters']} parameters"
        )
        self.logger_obj.info(
            f"Loss: {'Focal' if use_focal_loss else 'BCE'}, "
            f"Scheduler: {'CosineWarmup' if use_cosine_scheduler else 'ReduceLROnPlateau'}"
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        validate_config(self.config, self.logger_obj)

    def _setup_metrics(self) -> None:
        """Initialize torchmetrics for tracking performance."""
        metrics = setup_metrics(self.num_classes, train=True, validation=True, test=True)
        for name, metric in metrics.items():
            setattr(self, name, metric)

    def _setup_loss_function(self) -> None:
        """Setup loss function with enhanced options."""
        self.loss_fn = setup_loss_function(
            use_focal_loss=self.use_focal_loss,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            label_smoothing=self.label_smoothing,
            class_weights_tensor=self.class_weights_tensor,
            logger=self.logger_obj
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on task type."""
        return calculate_loss(self.loss_fn, logits, targets)

    def _get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to predictions."""
        return get_predictions(logits)

    def _prepare_targets_for_metrics(self, targets: torch.Tensor) -> torch.Tensor:
        """Prepare targets for metric computation."""
        return prepare_targets_for_metrics(targets)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform training step."""
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)

        preds = self._get_predictions(logits)
        targets_for_metrics = self._prepare_targets_for_metrics(y)

        self.train_accuracy.update(preds, targets_for_metrics)
        self.train_f1.update(preds, targets_for_metrics)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
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

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
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
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
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
                }
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
                    "strict": True
                }
            }

    def on_train_epoch_start(self) -> None:
        """Called at the start of each training epoch."""
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        current_lr = optimizer.param_groups[0]["lr"]  # type: ignore[index]
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
        self.logger_obj.info(f"Progressive unfreeze: {self.unfrozen_layers} layers unfrozen")

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        self.model.freeze_backbone()
        self.unfrozen_layers = 0

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        self.model.unfreeze_backbone()

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        return get_model_summary(
            model=self.model,
            config=self.config,
            lightning_module_name="LitResNetEnhanced",
            use_focal_loss=self.use_focal_loss,
            label_smoothing=self.label_smoothing,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma,
            unfrozen_layers=self.unfrozen_layers,
            device=self.device
        )

