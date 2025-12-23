"""
Enhanced training v2 - Adjusted hyperparameters for better recall.

Key changes from v1:
- focal_alpha=0.75 (more weight to positive class for better recall)
- Higher initial learning rate (0.001)
- Disabled label smoothing
- Longer patience (10 epochs)
- More epochs (30)
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet_enhanced import (
    LitResNetEnhanced,
    ProgressiveUnfreezeCallback,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.xray_data_module import XRayDataModule
from federated_pneumonia_detection.src.control.dl_model.utils import DataSourceExtractor
from federated_pneumonia_detection.src.utils.data_processing import (
    load_metadata,
    create_train_val_split,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedTrainerV2:
    """Enhanced training orchestrator v2 with recall-optimized settings."""

    def __init__(
        self,
        checkpoint_dir: str,
        logs_dir: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logs_dir = Path(logs_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.config = ConfigManager()
        self._apply_enhanced_config(config_overrides)
        self.data_extractor = DataSourceExtractor(logger)

    def _apply_enhanced_config(self, overrides: Optional[Dict[str, Any]] = None):
        """Apply recall-optimized configuration."""
        enhanced_defaults = {
            "system.sample_fraction": 1.0,
            "experiment.epochs": 30,  # More epochs
            "experiment.batch_size": 32,
            "experiment.learning_rate": 0.001,  # Higher LR
            "experiment.dropout_rate": 0.3,  # Less dropout
            "experiment.early_stopping_patience": 10,  # More patience
            "experiment.use_custom_preprocessing": True,
            "experiment.adaptive_histogram": True,
            "experiment.contrast_stretch": True,
            "experiment.fine_tune_layers_count": 0,
            "experiment.weight_decay": 0.0001,  # Less regularization
        }

        for key, value in enhanced_defaults.items():
            try:
                self.config.set(key, value)
            except Exception as e:
                logger.warning(f"Could not set {key}: {e}")

        if overrides:
            for key, value in overrides.items():
                try:
                    self.config.set(key, value)
                except Exception:
                    pass

        logger.info("Enhanced v2 configuration applied")

    def _compute_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        target_col = self.config.get("columns.target", "Target")
        y = np.array(train_df[target_col].values, dtype=int)
        unique_classes = np.unique(y)
        logger.info(f"Unique classes: {unique_classes}")

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=y
        )

        if len(class_weights) == 2:
            return torch.tensor(class_weights, dtype=torch.float32)
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    def train(
        self,
        source_path: str,
        experiment_name: str = "enhanced_v2",
        # Recall-optimized Focal Loss settings
        use_focal_loss: bool = True,
        focal_alpha: float = 0.75,  # MORE weight to positive class
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,  # Disabled
        use_cosine_scheduler: bool = True,
        use_progressive_unfreeze: bool = True,
        unfreeze_epochs: list = None,
    ) -> Dict[str, Any]:
        """Run recall-optimized training."""
        logger.info("=" * 60)
        logger.info("ENHANCED TRAINING V2 - Recall Optimized")
        logger.info("=" * 60)
        logger.info(f"Focal Alpha: {focal_alpha} (higher = more weight to positive)")
        logger.info(f"Label Smoothing: {label_smoothing}")

        # Step 1: Extract data
        logger.info("\nStep 1: Extracting data...")
        image_dir, csv_path = self.data_extractor.extract_and_validate(source_path)

        # Step 2: Prepare dataset
        logger.info("\nStep 2: Preparing dataset...")
        train_df, val_df = self._prepare_dataset(csv_path, image_dir)
        logger.info(f"Dataset: {len(train_df)} train, {len(val_df)} val")

        # Log class distribution
        target_col = self.config.get("columns.target", "Target")
        train_pos = (train_df[target_col] == 1).sum()
        train_neg = (train_df[target_col] == 0).sum()
        logger.info(f"Training class balance: {train_neg} negative, {train_pos} positive")
        logger.info(f"Positive ratio: {train_pos / len(train_df):.2%}")

        # Step 3: Class weights
        logger.info("\nStep 3: Computing class weights...")
        class_weights = self._compute_class_weights(train_df)
        logger.info(f"Class weights: {class_weights}")

        # Step 4: Data module
        logger.info("\nStep 4: Creating data module...")
        data_module = XRayDataModule(
            train_df=train_df,
            val_df=val_df,
            config=self.config,
            image_dir=image_dir,
            validate_images_on_init=False,
        )

        # Step 5: Enhanced model
        logger.info("\nStep 5: Creating enhanced model...")
        model = LitResNetEnhanced(
            config=self.config,
            class_weights_tensor=class_weights,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            use_cosine_scheduler=use_cosine_scheduler,
            monitor_metric="val_recall",
        )

        model_info = model.get_model_summary()
        logger.info(f"Model: {model_info['total_parameters']:,} parameters")

        # Step 6: Callbacks
        logger.info("\nStep 6: Setting up callbacks...")
        callbacks = self._create_callbacks(
            use_progressive_unfreeze=use_progressive_unfreeze,
            unfreeze_epochs=unfreeze_epochs,
        )

        # Step 7: Trainer
        logger.info("\nStep 7: Creating trainer...")
        trainer = self._create_trainer(callbacks, experiment_name)

        # Step 8: Train
        logger.info("\nStep 8: Starting training...")
        torch.set_float32_matmul_precision("high")
        trainer.fit(model, data_module)

        # Step 9: Results
        logger.info("\nStep 9: Collecting results...")
        results = self._collect_results(trainer, model)
        self._log_final_results(results)
        self._save_results(results, experiment_name)

        return results

    def _prepare_dataset(self, csv_path: str, image_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = load_metadata(csv_path, self.config, logger)
        target_col = self.config.get("columns.target", "Target")
        train_df, val_df = create_train_val_split(
            df,
            self.config.get("experiment.validation_split", 0.2),
            target_col,
            self.config.get("experiment.seed", 42),
            logger,
        )
        return train_df, val_df

    def _create_callbacks(
        self,
        use_progressive_unfreeze: bool = True,
        unfreeze_epochs: list = None,
    ) -> list:
        callbacks = []

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename="v2_{epoch:02d}_{val_recall:.3f}_{val_f1:.3f}",
            monitor="val_recall",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

        early_stopping = EarlyStopping(
            monitor="val_recall",
            mode="max",
            patience=self.config.get("experiment.early_stopping_patience", 10),
            min_delta=0.001,
            verbose=True,
        )
        callbacks.append(early_stopping)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        if use_progressive_unfreeze:
            total_epochs = self.config.get("experiment.epochs", 30)
            if unfreeze_epochs is None:
                unfreeze_epochs = [
                    int(total_epochs * 0.25),
                    int(total_epochs * 0.45),
                    int(total_epochs * 0.65),
                ]
            progressive_callback = ProgressiveUnfreezeCallback(
                unfreeze_epochs=unfreeze_epochs,
                layers_per_unfreeze=3,
            )
            callbacks.append(progressive_callback)
            logger.info(f"Progressive unfreezing at epochs: {unfreeze_epochs}")

        return callbacks

    def _create_trainer(self, callbacks: list, experiment_name: str) -> pl.Trainer:
        tb_logger = TensorBoardLogger(
            save_dir=str(self.logs_dir),
            name=experiment_name,
        )

        trainer = pl.Trainer(
            max_epochs=self.config.get("experiment.epochs", 30),
            accelerator="auto",
            devices=1,
            precision="16-mixed",
            callbacks=callbacks,
            logger=tb_logger,
            enable_progress_bar=True,
            log_every_n_steps=1,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            deterministic=False,
        )

        return trainer

    def _collect_results(self, trainer: pl.Trainer, model: LitResNetEnhanced) -> Dict[str, Any]:
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break

        best_model_path = None
        best_model_score = None
        if checkpoint_callback:
            best_model_path = checkpoint_callback.best_model_path
            if checkpoint_callback.best_model_score is not None:
                best_model_score = float(checkpoint_callback.best_model_score)

        return {
            "best_model_path": best_model_path,
            "best_model_score": best_model_score,
            "current_epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "model_summary": model.get_model_summary(),
            "configuration": {
                "epochs": self.config.get("experiment.epochs"),
                "batch_size": self.config.get("experiment.batch_size"),
                "learning_rate": self.config.get("experiment.learning_rate"),
                "weight_decay": self.config.get("experiment.weight_decay"),
                "dropout_rate": self.config.get("experiment.dropout_rate"),
                "use_focal_loss": model.use_focal_loss,
                "focal_alpha": model.focal_alpha,
                "focal_gamma": model.focal_gamma,
                "label_smoothing": model.label_smoothing,
                "use_cosine_scheduler": model.use_cosine_scheduler,
            },
        }

    def _log_final_results(self, results: Dict[str, Any]):
        logger.info("\n" + "=" * 60)
        logger.info("ENHANCED TRAINING V2 COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nBest val_recall: {results.get('best_model_score', 'N/A')}")
        logger.info(f"Best model: {results.get('best_model_path', 'N/A')}")
        logger.info(f"Total epochs: {results.get('current_epoch', 'N/A')}")

    def _save_results(self, results: Dict[str, Any], experiment_name: str):
        results_path = self.logs_dir / f"{experiment_name}_results.json"
        serializable = {}
        for key, value in results.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable[key] = value
            elif isinstance(value, dict):
                serializable[key] = {
                    k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                    for k, v in value.items()
                }
            else:
                serializable[key] = str(value)

        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"\nResults saved to: {results_path}")


def main():
    source_path = r"C:\Users\User\Projects\FYP2\Training_Sample_5pct.zip"
    output_dir = project_root / "model_enhancement_results" / "enhanced_v2"

    trainer = EnhancedTrainerV2(
        checkpoint_dir=str(output_dir / "checkpoints"),
        logs_dir=str(output_dir / "logs"),
    )

    results = trainer.train(
        source_path=source_path,
        experiment_name="enhanced_pneumonia_v2",
        use_focal_loss=True,
        focal_alpha=0.75,  # More weight to positive class
        focal_gamma=2.0,
        label_smoothing=0.0,  # Disabled
        use_cosine_scheduler=True,
        use_progressive_unfreeze=True,
    )

    return results


if __name__ == "__main__":
    main()
