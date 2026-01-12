"""
Checkpoint callback for tracking highest validation recall.
"""

import logging
import pytorch_lightning as pl


class HighestValRecallCallback(pl.Callback):
    """Custom callback to track highest validation recall achieved during training."""

    def __init__(self):
        super().__init__()
        self.best_recall = 0.0
        self.logger = logging.getLogger(__name__)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Track the highest validation recall."""
        current_recall = trainer.callback_metrics.get("val_recall", 0.0)
        if isinstance(current_recall, torch.Tensor):
            current_recall = current_recall.item()

        if current_recall > self.best_recall:
            self.best_recall = current_recall
            self.logger.info(f"New best validation recall: {self.best_recall:.4f}")
