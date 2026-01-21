"""
Progressive training callbacks for PyTorch Lightning.

Provides callbacks for progressive layer unfreezing during training.
"""

import logging
import pytorch_lightning as pl


class ProgressiveUnfreezeCallback(pl.Callback):
    """
    Callback for progressive unfreezing during training.

    Gradually unfreezes backbone layers as training progresses.
    """

    def __init__(
        self,
        unfreeze_epochs: list = None,
        layers_per_unfreeze: int = 2,
    ):
        """
        Initialize callback.

        Args:
            unfreeze_epochs: Epochs at which to unfreeze layers.
                             Default: [5, 10, 15] for a 20-epoch training.
            layers_per_unfreeze: Number of layers to unfreeze each time.
        """
        super().__init__()
        self.unfreeze_epochs = unfreeze_epochs or [5, 10, 15]
        self.layers_per_unfreeze = layers_per_unfreeze
        self.logger = logging.getLogger(__name__)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Check if we should unfreeze layers at this epoch."""
        current_epoch = trainer.current_epoch

        if current_epoch in self.unfreeze_epochs:
            if hasattr(pl_module, "progressive_unfreeze"):
                pl_module.progressive_unfreeze(self.layers_per_unfreeze)
                self.logger.info(
                    f"Epoch {current_epoch}: Unfroze {self.layers_per_unfreeze} more layers"
                )
