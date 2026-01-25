"""
Optimizer and scheduler factory for PyTorch Lightning modules.
"""

from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional

import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager


class OptimizerFactory:
    """
    Factory for creating optimizer and scheduler configurations.

    This class centralizes the logic for selecting and configuring optimizers
    and learning rate schedulers based on the system configuration.
    """

    @staticmethod
    def create_configuration(
        params: Iterator[nn.Parameter],
        config: "ConfigManager",
        monitor_metric: str = "val_recall",
        use_cosine_scheduler: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Create the optimizer and scheduler configuration.

        Args:
            params: Model parameters to optimize.
            config: Configuration manager instance.
            monitor_metric: Metric to monitor for scheduler (e.g., 'val_recall').
            use_cosine_scheduler: Whether to use cosine annealing scheduler.
                                 If None, uses value from config.

        Returns:
            A dictionary compatible with PyTorch Lightning's configure_optimizers.
        """
        # 1. Optimizer Selection
        optimizer_type = config.get("experiment.optimizer_type", "adamw").lower()
        learning_rate = config.get("experiment.learning_rate", 0.001)
        weight_decay = config.get("experiment.weight_decay", 0.0001)

        if optimizer_type == "adamw":
            optimizer = optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "adam":
            optimizer = optim.Adam(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            # Default to AdamW as per original implementation
            optimizer = optim.AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        # 2. Scheduler Selection
        # Check for explicit scheduler type or fall back to use_cosine_scheduler flag
        if use_cosine_scheduler is None:
            use_cosine_scheduler = config.get("experiment.use_cosine_scheduler", True)

        scheduler_type = config.get(
            "experiment.scheduler_type",
            "cosine" if use_cosine_scheduler else "plateau",
        ).lower()

        if scheduler_type == "cosine":
            total_epochs = config.get("experiment.epochs", 10)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(total_epochs // 3, 1),
                T_mult=1,
                eta_min=config.get("experiment.min_lr", 1e-7),
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
            # Default to ReduceLROnPlateau
            mode = "max" if "recall" in monitor_metric else "min"
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=config.get("experiment.reduce_lr_factor", 0.5),
                patience=config.get("experiment.reduce_lr_patience", 3),
                min_lr=config.get("experiment.min_lr", 1e-7),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor_metric,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                },
            }
