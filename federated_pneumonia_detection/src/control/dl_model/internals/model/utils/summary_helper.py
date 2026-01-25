"""
Model summary utility for LitResNetEnhanced.
"""

from typing import Any, Dict

import pytorch_lightning as pl


class SummaryHelper:
    """
    Utility for generating comprehensive model summaries.
    """

    @staticmethod
    def get_summary(pl_module: pl.LightningModule) -> Dict[str, Any]:
        """
        Get comprehensive model summary.

        Args:
            pl_module: The LitResNetEnhanced instance

        Returns:
            Dictionary containing model information and hyperparameters
        """
        # Accessing protected attributes/methods of pl_module for summary
        model_info = pl_module.model.get_model_info()
        model_info.update(
            {
                "lightning_module": "LitResNetEnhanced",
                "optimizer": "AdamW",
                "learning_rate": pl_module.config.get("experiment.learning_rate", 0),
                "weight_decay": pl_module.config.get("experiment.weight_decay", 0),
                "scheduler": "CosineAnnealingWarmRestarts"
                if getattr(pl_module, "use_cosine_scheduler", False)
                else "ReduceLROnPlateau",
                "loss_function": "FocalLoss"
                if getattr(pl_module, "use_focal_loss", False)
                else "BCEWithLogitsLoss",
                "label_smoothing": getattr(pl_module, "label_smoothing", 0.0),
                "focal_alpha": getattr(pl_module, "focal_alpha", None)
                if getattr(pl_module, "use_focal_loss", False)
                else None,
                "focal_gamma": getattr(pl_module, "focal_gamma", None)
                if getattr(pl_module, "use_focal_loss", False)
                else None,
                "unfrozen_layers": getattr(pl_module, "unfrozen_layers", 0),
                "device": str(pl_module.device),
            },
        )
        return model_info
