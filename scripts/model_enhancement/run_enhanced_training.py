"""
Enhanced training script for high-accuracy pneumonia detection.

Implements:
- Full dataset usage (no subsampling)
- Focal Loss for class imbalance
- Progressive unfreezing of backbone
- Cosine Annealing with Warmup
- Label smoothing
- CLAHE preprocessing for X-rays
- Extended training epochs
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


class EnhancedTrainer:
    """
    Enhanced training orchestrator for high-accuracy pneumonia detection.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        logs_dir: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize enhanced trainer.

        Args:
            checkpoint_dir: Directory for model checkpoints
            logs_dir: Directory for training logs
            config_overrides: Optional configuration overrides
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logs_dir = Path(logs_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration with overrides
        self.config = ConfigManager()
        self._apply_enhanced_config(config_overrides)

        # Initialize data extractor
        self.data_extractor = DataSourceExtractor(logger)

    def _apply_enhanced_config(self, overrides: Optional[Dict[str, Any]] = None):
        """Apply enhanced configuration settings."""
        # Enhanced defaults for high accuracy
        enhanced_defaults = {
            "system.sample_fraction": 1.0,  # Use full dataset
            "experiment.epochs": 25,  # More epochs
            "experiment.batch_size": 32,  # Keep reasonable batch size
            "experiment.learning_rate": 0.0003,  # Lower LR for fine-tuning
            "experiment.dropout_rate": 0.4,  # Slightly more dropout
            "experiment.early_stopping_patience": 8,  # More patience
            "experiment.use_custom_preprocessing": True,  # Enable CLAHE
            "experiment.adaptive_histogram": True,  # Enable CLAHE
            "experiment.contrast_stretch": True,  # Enable contrast stretch
            "experiment.fine_tune_layers_count": 0,  # Start frozen
            "experiment.weight_decay": 0.0005,  # More regularization
        }

        # Apply enhanced defaults
        for key, value in enhanced_defaults.items():
            try:
                self.config.set(key, value)
            except Exception as e:
                logger.warning(f"Could not set {key}: {e}")

        # Apply user overrides
        if overrides:
            for key, value in overrides.items():
                try:
                    self.config.set(key, value)
                except Exception as e:
                    logger.warning(f"Could not set override {key}: {e}")

        logger.info("Enhanced configuration applied")

    def _compute_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        """Compute class weights from training data."""
        target_col = self.config.get("columns.target", "Target")
        y = train_df[target_col].values

        # Ensure integer labels
        y = np.array(y, dtype=int)

        # Get unique classes
        unique_classes = np.unique(y)
        logger.info(f"Unique classes in target: {unique_classes}")

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=y
        )

        # Ensure we have weights for classes 0 and 1
        if len(class_weights) == 2:
            return torch.tensor(class_weights, dtype=torch.float32)
        else:
            # If only one class present, return balanced weights
            logger.warning(f"Only {len(unique_classes)} classes found, using default weights")
            return torch.tensor([1.0, 1.0], dtype=torch.float32)

    def train(
        self,
        source_path: str,
        experiment_name: str = "enhanced_training",
        # Enhanced training options
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        use_cosine_scheduler: bool = True,
        use_progressive_unfreeze: bool = True,
        unfreeze_epochs: list = None,
    ) -> Dict[str, Any]:
        """
        Run enhanced training.

        Args:
            source_path: Path to dataset (zip or directory)
            experiment_name: Name for experiment logging
            use_focal_loss: Use Focal Loss for hard examples
            focal_alpha: Alpha parameter for Focal Loss
            focal_gamma: Gamma parameter for Focal Loss
            label_smoothing: Label smoothing factor
            use_cosine_scheduler: Use cosine annealing scheduler
            use_progressive_unfreeze: Enable progressive unfreezing
            unfreeze_epochs: Epochs at which to unfreeze layers

        Returns:
            Training results dictionary
        """
        logger.info("=" * 60)
        logger.info("ENHANCED TRAINING - High Accuracy Configuration")
        logger.info("=" * 60)

        # Step 1: Extract and validate data
        logger.info("\nStep 1: Extracting and validating data...")
        image_dir, csv_path = self.data_extractor.extract_and_validate(source_path)

        # Step 2: Prepare dataset (NO SUBSAMPLING)
        logger.info("\nStep 2: Preparing dataset (full data)...")
        train_df, val_df = self._prepare_dataset(csv_path, image_dir)
        logger.info(f"Dataset: {len(train_df)} training, {len(val_df)} validation samples")

        # Step 3: Compute class weights
        logger.info("\nStep 3: Computing class weights...")
        class_weights = self._compute_class_weights(train_df)
        logger.info(f"Class weights: {class_weights}")

        # Step 4: Create data module
        logger.info("\nStep 4: Creating data module...")
        data_module = XRayDataModule(
            train_df=train_df,
            val_df=val_df,
            config=self.config,
            image_dir=image_dir,
            validate_images_on_init=False,
        )

        # Step 5: Create enhanced model
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

        # Log model info
        model_info = model.get_model_summary()
        logger.info(f"Model: {model_info['total_parameters']:,} total parameters")
        logger.info(f"Trainable: {model_info['trainable_parameters']:,} parameters")
        logger.info(f"Loss: {model_info['loss_function']}")
        logger.info(f"Scheduler: {model_info['scheduler']}")

        # Step 6: Setup callbacks
        logger.info("\nStep 6: Setting up callbacks...")
        callbacks = self._create_callbacks(
            use_progressive_unfreeze=use_progressive_unfreeze,
            unfreeze_epochs=unfreeze_epochs,
        )

        # Step 7: Create trainer
        logger.info("\nStep 7: Creating PyTorch Lightning trainer...")
        trainer = self._create_trainer(callbacks, experiment_name)

        # Step 8: Train
        logger.info("\nStep 8: Starting training...")
        logger.info(f"Training for {self.config.get('experiment.epochs')} epochs")
        logger.info(f"Batch size: {self.config.get('experiment.batch_size')}")
        logger.info(f"Learning rate: {self.config.get('experiment.learning_rate')}")

        # Set precision for tensor cores
        torch.set_float32_matmul_precision("high")

        trainer.fit(model, data_module)

        # Step 9: Collect results
        logger.info("\nStep 9: Collecting results...")
        results = self._collect_results(trainer, model)

        self._log_final_results(results)
        self._save_results(results, experiment_name)

        return results

    def _prepare_dataset(self, csv_path: str, image_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and validation datasets."""
        # Load metadata
        df = load_metadata(csv_path, self.config, logger)
        logger.info(f"Loaded {len(df)} samples from metadata")

        # NO SUBSAMPLING - use full dataset
        target_col = self.config.get("columns.target", "Target")

        # Create train/val split
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
        """Create training callbacks."""
        callbacks = []

        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename="enhanced_{epoch:02d}_{val_recall:.3f}_{val_f1:.3f}",
            monitor="val_recall",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_recall",
            mode="max",
            patience=self.config.get("experiment.early_stopping_patience", 8),
            min_delta=0.001,
            verbose=True,
        )
        callbacks.append(early_stopping)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # Progressive unfreezing
        if use_progressive_unfreeze:
            total_epochs = self.config.get("experiment.epochs", 25)
            if unfreeze_epochs is None:
                # Unfreeze at 30%, 50%, 70% of training
                unfreeze_epochs = [
                    int(total_epochs * 0.3),
                    int(total_epochs * 0.5),
                    int(total_epochs * 0.7),
                ]
            progressive_callback = ProgressiveUnfreezeCallback(
                unfreeze_epochs=unfreeze_epochs,
                layers_per_unfreeze=3,
            )
            callbacks.append(progressive_callback)
            logger.info(f"Progressive unfreezing at epochs: {unfreeze_epochs}")

        return callbacks

    def _create_trainer(self, callbacks: list, experiment_name: str) -> pl.Trainer:
        """Create PyTorch Lightning trainer."""
        tb_logger = TensorBoardLogger(
            save_dir=str(self.logs_dir),
            name=experiment_name,
        )

        trainer = pl.Trainer(
            max_epochs=self.config.get("experiment.epochs", 25),
            accelerator="auto",
            devices=1,
            precision="16-mixed",  # Use mixed precision for speed
            callbacks=callbacks,
            logger=tb_logger,
            enable_progress_bar=True,
            log_every_n_steps=1,
            gradient_clip_val=1.0,  # Gradient clipping
            accumulate_grad_batches=1,
            deterministic=False,  # Faster training
        )

        return trainer

    def _collect_results(self, trainer: pl.Trainer, model: LitResNetEnhanced) -> Dict[str, Any]:
        """Collect training results."""
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

        # Get logged metrics
        metrics_history = []
        if hasattr(trainer, "logged_metrics"):
            metrics_history.append(dict(trainer.logged_metrics))

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
        """Log final training results."""
        logger.info("\n" + "=" * 60)
        logger.info("ENHANCED TRAINING COMPLETE")
        logger.info("=" * 60)

        logger.info(f"\nBest Model Score (val_recall): {results.get('best_model_score', 'N/A')}")
        logger.info(f"Best Model Path: {results.get('best_model_path', 'N/A')}")
        logger.info(f"Total Epochs: {results.get('current_epoch', 'N/A')}")

        config = results.get("configuration", {})
        logger.info("\nConfiguration Used:")
        for key, value in config.items():
            logger.info(f"  - {key}: {value}")

    def _save_results(self, results: Dict[str, Any], experiment_name: str):
        """Save results to JSON file."""
        results_path = self.logs_dir / f"{experiment_name}_results.json"

        # Make JSON serializable
        serializable = {}
        for key, value in results.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable[key] = value
            elif isinstance(value, dict):
                serializable[key] = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                                     for k, v in value.items()}
            else:
                serializable[key] = str(value)

        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"\nResults saved to: {results_path}")


def main():
    """Main entry point for enhanced training."""
    # Configuration
    source_path = r"C:\Users\User\Projects\FYP2\Training_Sample_5pct.zip"
    output_dir = project_root / "model_enhancement_results" / "enhanced_v1"

    # Create trainer
    trainer = EnhancedTrainer(
        checkpoint_dir=str(output_dir / "checkpoints"),
        logs_dir=str(output_dir / "logs"),
    )

    # Run enhanced training
    results = trainer.train(
        source_path=source_path,
        experiment_name="enhanced_pneumonia_v1",
        # Enhanced options
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        label_smoothing=0.1,
        use_cosine_scheduler=True,
        use_progressive_unfreeze=True,
    )

    return results


if __name__ == "__main__":
    main()
