import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import pytorch_lightning as pl
import torch
from torchvision.models import ResNet50_Weights
from federated_pneumonia_detection.src.control.dl_model.internals.model.optimizers.factory import (
    OptimizerFactory,
)
from federated_pneumonia_detection.src.entities.resnet_with_custom_head import (
    ResNetWithCustomHead,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.utils.loss_factory import (
    LossFactory,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.utils.metrics_handler import (
    MetricsHandler,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.utils.step_logic import (
    StepLogic,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.utils.summary_helper import (
    SummaryHelper,
)


class LitResNetEnhanced(pl.LightningModule):
    model: ResNetWithCustomHead
    config: Any

    def __init__(
        self,
        config: Optional[Any] = None,
        base_model_weights: Optional[ResNet50_Weights] = None,
        class_weights_tensor: Optional[torch.Tensor] = None,
        num_classes: int = 1,
        monitor_metric: str = "val_recall",
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        use_cosine_scheduler: bool = True,
        warmup_epochs: int = 3,
    ):
        super().__init__()
        if config is None:
            from federated_pneumonia_detection.config.config_manager import (
                ConfigManager,
            )

            config = ConfigManager()
        self.config, self.num_classes, self.monitor_metric, self.logger_obj = (
            config,
            num_classes,
            monitor_metric,
            logging.getLogger(__name__),
        )
        (
            self.use_focal_loss,
            self.focal_alpha,
            self.focal_gamma,
            self.label_smoothing,
        ) = use_focal_loss, focal_alpha, focal_gamma, label_smoothing
        self.use_cosine_scheduler, self.warmup_epochs, self.unfrozen_layers = (
            use_cosine_scheduler,
            warmup_epochs,
            0,
        )
        self.save_hyperparameters(
            ignore=["config", "base_model_weights", "class_weights_tensor"]
        )
        if self.config.get("experiment.learning_rate", 0) <= 0:
            raise ValueError("Learning rate must be positive")
        if self.config.get("experiment.weight_decay", -1) < 0:
            raise ValueError("Weight decay must be non-negative")
        self.model = ResNetWithCustomHead(
            config=self.config,
            base_model_weights=base_model_weights,
            num_classes=num_classes,
            dropout_rate=self.config.get("experiment.dropout_rate", 0.5),
            fine_tune_layers_count=self.config.get(
                "experiment.fine_tune_layers_count", 0
            ),
        )
        if self.config.get("experiment.use_torch_compile", False):
            self.model = cast(
                ResNetWithCustomHead,
                torch.compile(
                    self.model,
                    mode=self.config.get("experiment.torch_compile_mode", "default"),
                ),
            )
        self.metrics_handler, self.step_logic = (
            MetricsHandler(num_classes),
            StepLogic(MetricsHandler(num_classes)),
        )
        self.step_logic.metrics_handler = self.metrics_handler
        self.loss_factory = LossFactory.create_loss_function(
            class_weights_tensor,
            use_focal_loss,
            focal_alpha,
            focal_gamma,
            label_smoothing,
            self.logger_obj,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _calculate_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return LossFactory.calculate_loss(self.loss_factory, logits, targets)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step_logic.execute_step(self, batch, "train")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step_logic.execute_step(self, batch, "val")

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step_logic.execute_step(self, batch, "test")

    def configure_optimizers(self) -> Any:
        return OptimizerFactory.create_configuration(
            self.parameters(),
            self.config,
            self.monitor_metric,
            use_cosine_scheduler=self.use_cosine_scheduler,
        )

    def on_train_epoch_start(self) -> None:
        opt = self.optimizers()
        optimizer = opt[0] if isinstance(opt, list) else opt
        lr = cast(Any, optimizer).param_groups[0]["lr"]
        self.log("learning_rate", lr, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        [
            self.log(n, v, on_epoch=True, sync_dist=True)
            for n, v in self.metrics_handler.get_confusion_matrix_metrics().items()
        ]
        self.metrics_handler.reset_confusion_matrix()

    def progressive_unfreeze(self, layers_to_unfreeze: int = 1) -> None:
        self.unfrozen_layers += layers_to_unfreeze
        self.model._unfreeze_last_n_layers(self.unfrozen_layers)

    def freeze_backbone(self) -> None:
        self.model.freeze_backbone()
        self.unfrozen_layers = 0

    def unfreeze_backbone(self) -> None:
        self.model.unfreeze_backbone()

    def _get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert logits to probabilities based on the number of classes.
        Used by server_evaluation.py.
        """
        if self.num_classes == 1:
            return torch.sigmoid(logits).flatten()
        return torch.softmax(logits, dim=1)

    def _prepare_targets_for_metrics(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Prepare targets for metric calculation.
        Used by server_evaluation.py.
        """
        # Pass-through as per requirement, but can be extended if casting is needed
        return targets

    def get_model_summary(self) -> Dict[str, Any]:
        return SummaryHelper.get_summary(self)
