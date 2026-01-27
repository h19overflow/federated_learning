"""
Gradient monitoring callback for tracking gradient norms and learning rate.
Provides insights into optimizer behavior and gradient flow.
"""

import logging
from typing import Dict, Optional

import pytorch_lightning as pl


class GradientMonitorCallback(pl.Callback):
    """
    Callback to monitor gradient norms and learning rate during training.

    Computes gradient statistics at configurable intervals to track
    gradient flow and optimizer state without impacting training performance.
    """

    def __init__(self, websocket_sender, sample_interval: int = 20):
        """
        Initialize gradient monitor callback.

        Args:
            websocket_sender: MetricsWebSocketSender instance for sending metrics
            sample_interval: Compute gradient stats every N steps (default: 20)
        """
        super().__init__()
        self.websocket_sender = websocket_sender
        self.sample_interval = sample_interval
        self.logger = logging.getLogger(__name__)
        self.step_counter = 0
        self.last_lr = None

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer,
    ) -> None:
        """
        Called before optimizer step to compute gradient statistics.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
            optimizer: Current optimizer
        """
        self.step_counter += 1

        # Sample every Nth step
        if self.step_counter % self.sample_interval != 0:
            return

        # Compute gradient norms
        layer_norms = self._compute_layer_norms(pl_module)

        if not layer_norms:
            return

        # Calculate aggregate statistics
        norm_values = list(layer_norms.values())
        total_norm = sum(norm_values)
        max_norm = max(norm_values)
        min_norm = min(norm_values)

        # Send gradient statistics
        if self.websocket_sender:
            self.websocket_sender.send_gradient_stats(
                step=trainer.global_step,
                total_norm=total_norm,
                layer_norms=layer_norms,
                max_norm=max_norm,
                min_norm=min_norm,
            )

        # Extract and send learning rate
        current_lr = self._get_learning_rate(optimizer)
        if current_lr is not None:
            # Only send LR update if it changed or this is first send
            if self.last_lr is None or abs(current_lr - self.last_lr) > 1e-9:
                scheduler_type = self._get_scheduler_type(trainer)
                if self.websocket_sender:
                    self.websocket_sender.send_lr_update(
                        current_lr=current_lr,
                        step=trainer.global_step,
                        epoch=trainer.current_epoch,
                        scheduler_type=scheduler_type,
                    )
                self.last_lr = current_lr

        self.logger.debug(
            f"[GradientMonitor] Step {trainer.global_step}: "
            f"total_norm={total_norm:.4f}, lr={current_lr}",
        )

    def _compute_layer_norms(self, pl_module: pl.LightningModule) -> Dict[str, float]:
        """
        Compute gradient norms per layer group.

        Args:
            pl_module: Lightning module

        Returns:
            Dictionary mapping layer prefixes to gradient norms
        """
        layer_norms = {}

        for name, param in pl_module.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue

            # Compute gradient norm for this parameter
            grad_norm = param.grad.data.norm(2).item()

            # Extract layer prefix (e.g., "encoder.layer1" from "encoder.layer1.conv.weight")  # noqa: E501
            layer_prefix = self._get_layer_prefix(name)

            # Aggregate norms by layer prefix
            if layer_prefix in layer_norms:
                layer_norms[layer_prefix] += grad_norm
            else:
                layer_norms[layer_prefix] = grad_norm

        return layer_norms

    def _get_layer_prefix(self, param_name: str) -> str:
        """
        Extract layer prefix from parameter name.

        Args:
            param_name: Full parameter name (e.g., "model.encoder.layer1.conv.weight")

        Returns:
            Layer prefix (e.g., "model.encoder.layer1")
        """
        # Split by dots and take first 3 parts (typically model.component.layer)
        parts = param_name.split(".")
        if len(parts) >= 3:
            return ".".join(parts[:3])
        elif len(parts) >= 2:
            return ".".join(parts[:2])
        else:
            return parts[0] if parts else "unknown"

    def _get_learning_rate(self, optimizer) -> Optional[float]:
        """
        Extract current learning rate from optimizer.

        Args:
            optimizer: PyTorch optimizer

        Returns:
            Current learning rate or None if not available
        """
        if hasattr(optimizer, "param_groups") and optimizer.param_groups:
            return optimizer.param_groups[0].get("lr")
        return None

    def _get_scheduler_type(self, trainer: pl.Trainer) -> Optional[str]:
        """
        Determine scheduler type from trainer configuration.

        Args:
            trainer: PyTorch Lightning trainer

        Returns:
            Scheduler type name or None
        """
        if hasattr(trainer, "lr_scheduler_configs") and trainer.lr_scheduler_configs:
            scheduler_config = trainer.lr_scheduler_configs[0]
            scheduler = scheduler_config.scheduler
            return scheduler.__class__.__name__ if scheduler else None
        return None
