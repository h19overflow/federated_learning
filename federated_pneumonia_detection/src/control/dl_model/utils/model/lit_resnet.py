"""
PyTorch Lightning module for ResNet-based pneumonia classification.
Provides comprehensive training, validation, and metrics tracking with configurable optimization.
"""

import logging
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
from torchvision.models import ResNet50_Weights

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.entities.resnet_with_custom_head import ResNetWithCustomHead


class LitResNet(pl.LightningModule):
    """
    PyTorch Lightning module for ResNet-based binary classification.

    Integrates ResNetWithCustomHead with comprehensive training pipeline,
    metrics tracking, and configurable optimization strategies.
    """

    def __init__(
        self,
        constants: SystemConstants,
        config: ExperimentConfig,
        base_model_weights: Optional[ResNet50_Weights] = None,
        class_weights_tensor: Optional[torch.Tensor] = None,
        num_classes: int = 1,
        monitor_metric: str = "val_loss"
    ):
        """
        Initialize Lightning module.

        Args:
            constants: SystemConstants for configuration
            config: ExperimentConfig for training parameters
            base_model_weights: Optional ResNet50 weights
            class_weights_tensor: Optional class weights for loss calculation
            num_classes: Number of output classes (1 for binary)
            monitor_metric: Metric to monitor for learning rate scheduling

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__()

        self.constants = constants
        self.config = config
        self.num_classes = num_classes
        self.monitor_metric = monitor_metric
        self.logger_obj = logging.getLogger(__name__)

        # Save hyperparameters (excluding non-serializable objects)
        self.save_hyperparameters(ignore=[
            "constants", "config", "base_model_weights", "class_weights_tensor"
        ])

        # Validate configuration
        self._validate_config()

        # Initialize model
        self.model = ResNetWithCustomHead(
            constants=constants,
            config=config,
            base_model_weights=base_model_weights,
            num_classes=num_classes,
            dropout_rate=getattr(config, 'dropout_rate', 0.5),
            fine_tune_layers_count=getattr(config, 'fine_tune_layers_count', 0)
        )

        # Store class weights
        self.class_weights_tensor = class_weights_tensor

        # Initialize metrics
        self._setup_metrics()

        # Setup loss function
        self._setup_loss_function()

        self.logger_obj.info(f"LitResNet initialized with {self.model.get_model_info()['total_parameters']} parameters")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.config.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")

        if not hasattr(self.config, 'early_stopping_patience'):
            self.logger_obj.warning("early_stopping_patience not found in config, using default")

    def _setup_metrics(self) -> None:
        """Initialize torchmetrics for tracking performance."""
        num_classes = 2 if self.num_classes == 1 else self.num_classes
        task_type = "binary" if self.num_classes == 1 else "multiclass"

        # Training metrics
        self.train_accuracy = torchmetrics.Accuracy(task=task_type, num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task=task_type, num_classes=num_classes)

        # Validation metrics
        self.val_accuracy = torchmetrics.Accuracy(task=task_type, num_classes=num_classes)
        self.val_precision = torchmetrics.Precision(task=task_type, num_classes=num_classes)
        self.val_recall = torchmetrics.Recall(task=task_type, num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task=task_type, num_classes=num_classes)
        self.val_auroc = torchmetrics.AUROC(task=task_type, num_classes=num_classes)

        # Test metrics (same as validation)
        self.test_accuracy = torchmetrics.Accuracy(task=task_type, num_classes=num_classes)
        self.test_precision = torchmetrics.Precision(task=task_type, num_classes=num_classes)
        self.test_recall = torchmetrics.Recall(task=task_type, num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task=task_type, num_classes=num_classes)
        self.test_auroc = torchmetrics.AUROC(task=task_type, num_classes=num_classes)

    def _setup_loss_function(self) -> None:
        """Setup loss function with optional class weighting."""
        if self.num_classes == 1:
            # Binary classification
            pos_weight = None
            if self.class_weights_tensor is not None:
                pos_weight = self.class_weights_tensor[1] / (self.class_weights_tensor[0] + 1e-8)
                self.logger_obj.info(f"Using positive class weight: {pos_weight}")

            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            # Multi-class classification
            weight = self.class_weights_tensor
            self.loss_fn = nn.CrossEntropyLoss(weight=weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate loss based on task type."""
        if self.num_classes == 1:
            # Binary classification - ensure targets are float and properly shaped
            targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()
            return self.loss_fn(logits, targets)
        else:
            # Multi-class classification - targets should be long integers
            targets = targets.long()
            return self.loss_fn(logits, targets)

    def _get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to predictions based on task type."""
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=1)

    def _prepare_targets_for_metrics(self, targets: torch.Tensor) -> torch.Tensor:
        """Prepare targets for metric computation."""
        if self.num_classes == 1:
            return targets.int().unsqueeze(1) if targets.dim() == 1 else targets.int()
        else:
            return targets.long()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform training step."""
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)

        # Get predictions for metrics
        preds = self._get_predictions(logits)
        targets_for_metrics = self._prepare_targets_for_metrics(y)

        # Update metrics
        self.train_accuracy.update(preds, targets_for_metrics)
        self.train_f1.update(preds, targets_for_metrics)

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform validation step."""
        x, y = batch
        logits = self(x)
        loss = self._calculate_loss(logits, y)

        # Get predictions for metrics
        preds = self._get_predictions(logits)
        targets_for_metrics = self._prepare_targets_for_metrics(y)

        # Update all validation metrics
        self.val_accuracy.update(preds, targets_for_metrics)
        self.val_precision.update(preds, targets_for_metrics)
        self.val_recall.update(preds, targets_for_metrics)
        self.val_f1.update(preds, targets_for_metrics)
        self.val_auroc.update(preds, targets_for_metrics)

        # Log metrics
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

        # Get predictions for metrics
        preds = self._get_predictions(logits)
        targets_for_metrics = self._prepare_targets_for_metrics(y)

        # Update test metrics
        self.test_accuracy.update(preds, targets_for_metrics)
        self.test_precision.update(preds, targets_for_metrics)
        self.test_recall.update(preds, targets_for_metrics)
        self.test_f1.update(preds, targets_for_metrics)
        self.test_auroc.update(preds, targets_for_metrics)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_auroc", self.test_auroc, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def predict_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        """Perform prediction step."""
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = self(x)
        predictions = self._get_predictions(logits)
        return predictions


    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Setup optimizer
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,

        )

        # Setup learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min' if 'loss' in self.monitor_metric else 'max',
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
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

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", current_lr, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Optional: Add custom validation epoch end logic
        pass

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        model_info = self.model.get_model_info()

        # Add Lightning-specific info
        model_info.update({
            'lightning_module': 'LitResNet',
            'optimizer': 'AdamW',
            'learning_rate': self.config.learning_rate,
            'weight_decay': self.config.weight_decay,
            'scheduler': 'ReduceLROnPlateau',
            'monitor_metric': self.monitor_metric,
            'class_weighted_loss': self.class_weights_tensor is not None,
            'device': str(self.device)
        })

        return model_info

    def freeze_backbone(self) -> None:
        """Freeze the backbone parameters."""
        self.model.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone parameters."""
        self.model.unfreeze_backbone()

    def set_fine_tuning_mode(self, enabled: bool = True) -> None:
        """Enable or disable fine-tuning mode."""
        if enabled:
            self.unfreeze_backbone()
            # Optionally reduce learning rate for fine-tuning
            for param_group in self.optimizers().param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
            self.logger_obj.info("Fine-tuning mode enabled, learning rate reduced")
        else:
            self.freeze_backbone()
            self.logger_obj.info("Fine-tuning mode disabled")

    def get_feature_maps(self, x: torch.Tensor, layer_name: Optional[str] = None) -> torch.Tensor:
        """Extract feature maps from model."""
        return self.model.get_feature_maps(x, layer_name)

    def compute_class_weights(self, train_dataloader) -> torch.Tensor:
        """
        Compute class weights from training data.

        Args:
            train_dataloader: Training DataLoader

        Returns:
            Class weights tensor
        """
        class_counts = torch.zeros(2 if self.num_classes == 1 else self.num_classes)

        for batch in train_dataloader:
            _, labels = batch
            if self.num_classes == 1:
                # Binary classification
                labels = labels.int()
                class_counts[0] += (labels == 0).sum()
                class_counts[1] += (labels == 1).sum()
            else:
                # Multi-class
                for i in range(self.num_classes):
                    class_counts[i] += (labels == i).sum()

        # Compute inverse frequency weights
        total_samples = class_counts.sum()
        class_weights = total_samples / (len(class_counts) * class_counts)

        self.logger_obj.info(f"Computed class weights: {class_weights}")
        return class_weights