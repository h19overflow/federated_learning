"""
Train on the FULL dataset (30,000 samples) using the enhanced model.
Uses the same high-recall configuration that achieved 97.8% recall on the sample.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from sklearn.utils.class_weight import compute_class_weight

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet_enhanced import (
    LitResNetEnhanced,
    ProgressiveUnfreezeCallback,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.xray_data_module import XRayDataModule
from federated_pneumonia_detection.src.utils.data_processing import load_metadata, create_train_val_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FullDatasetTrainer:
    """Train on the full 30k dataset with high-recall configuration."""

    def __init__(self, checkpoint_dir: str, logs_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logs_dir = Path(logs_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.config = ConfigManager()
        self._apply_full_dataset_config()

    def _apply_full_dataset_config(self):
        """Configuration optimized for full dataset training."""
        config_settings = {
            "system.sample_fraction": 1.0,  # Use ALL data
            "experiment.epochs": 20,  # Fewer epochs needed with more data
            "experiment.batch_size": 64,  # Larger batch for stability
            "experiment.learning_rate": 0.001,
            "experiment.dropout_rate": 0.3,
            "experiment.early_stopping_patience": 5,
            "experiment.fine_tune_layers_count": 0,
            "experiment.weight_decay": 0.0001,
            "experiment.validation_split": 0.15,  # 15% validation
        }
        for key, value in config_settings.items():
            try:
                self.config.set(key, value)
            except Exception:
                pass
        logger.info("Full dataset configuration applied")

    def _compute_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        target_col = self.config.get("columns.target", "Target")
        y = np.array(train_df[target_col].values, dtype=int)
        unique_classes = np.unique(y)
        class_weights = compute_class_weight("balanced", classes=unique_classes, y=y)
        return torch.tensor(class_weights, dtype=torch.float32)

    def train(
        self,
        data_dir: str,
        experiment_name: str = "full_dataset_training",
        focal_alpha: float = 0.75,  # High recall config
        focal_gamma: float = 2.0,
    ) -> Dict[str, Any]:
        """Train on full dataset."""
        logger.info("=" * 70)
        logger.info("FULL DATASET TRAINING (~30,000 samples)")
        logger.info("=" * 70)

        data_path = Path(data_dir)
        csv_path = data_path / "stage2_train_metadata.csv"
        image_dir = data_path / "Images"

        # Load data
        logger.info("\n[1/6] Loading full dataset...")
        df = load_metadata(str(csv_path), self.config, logger)
        logger.info(f"Total samples: {len(df)}")

        # Class distribution
        target_col = self.config.get("columns.target", "Target")
        pos_count = (df[target_col] == 1).sum()
        neg_count = (df[target_col] == 0).sum()
        logger.info(f"Class distribution: {neg_count} negative, {pos_count} positive ({pos_count/len(df):.1%})")

        # Train/val split
        logger.info("\n[2/6] Creating train/val split...")
        train_df, val_df = create_train_val_split(
            df,
            self.config.get("experiment.validation_split", 0.15),
            target_col,
            self.config.get("experiment.seed", 42),
            logger,
        )
        logger.info(f"Train: {len(train_df)}, Validation: {len(val_df)}")

        # Class weights
        logger.info("\n[3/6] Computing class weights...")
        class_weights = self._compute_class_weights(train_df)
        logger.info(f"Class weights: {class_weights}")

        # Data module
        logger.info("\n[4/6] Creating data module...")
        data_module = XRayDataModule(
            train_df=train_df,
            val_df=val_df,
            config=self.config,
            image_dir=str(image_dir),
            validate_images_on_init=False,
        )

        # Model
        logger.info("\n[5/6] Creating model...")
        model = LitResNetEnhanced(
            config=self.config,
            class_weights_tensor=class_weights,
            use_focal_loss=True,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            label_smoothing=0.0,
            use_cosine_scheduler=True,
            monitor_metric="val_recall",
        )
        logger.info(f"Parameters: {model.get_model_summary()['total_parameters']:,}")

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=str(self.checkpoint_dir),
                filename="full_{epoch:02d}_{val_recall:.3f}_{val_f1:.3f}",
                monitor="val_recall",
                mode="max",
                save_top_k=3,
                save_last=True,
                verbose=True,
            ),
            EarlyStopping(
                monitor="val_recall",
                mode="max",
                patience=self.config.get("experiment.early_stopping_patience", 5),
                min_delta=0.001,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            ProgressiveUnfreezeCallback(
                unfreeze_epochs=[5, 10, 15],
                layers_per_unfreeze=3,
            ),
        ]

        # Trainer
        logger.info("\n[6/6] Starting training...")
        trainer = pl.Trainer(
            max_epochs=self.config.get("experiment.epochs", 20),
            accelerator="auto",
            devices=1,
            precision="16-mixed",
            callbacks=callbacks,
            logger=[
                TensorBoardLogger(str(self.logs_dir), name=experiment_name),
                CSVLogger(str(self.logs_dir), name=f"{experiment_name}_csv"),
            ],
            enable_progress_bar=True,
            log_every_n_steps=50,
            gradient_clip_val=1.0,
        )

        torch.set_float32_matmul_precision("high")
        trainer.fit(model, data_module)

        # Results
        best_score = None
        best_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_score is not None:
                best_score = float(cb.best_model_score)
                best_path = cb.best_model_path
                break

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Best val_recall: {best_score}")
        logger.info(f"Best model: {best_path}")

        results = {
            "best_recall": best_score,
            "best_model_path": best_path,
            "total_epochs": trainer.current_epoch,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
        }

        # Save results
        with open(self.logs_dir / f"{experiment_name}_results.json", "w") as f:
            json.dump(results, f, indent=2)

        return results


def main():
    data_dir = r"C:\Users\User\Projects\FYP2\training_data"
    output_dir = project_root / "model_enhancement_results" / "full_dataset"

    trainer = FullDatasetTrainer(
        checkpoint_dir=str(output_dir / "checkpoints"),
        logs_dir=str(output_dir / "logs"),
    )

    results = trainer.train(
        data_dir=data_dir,
        experiment_name="full_dataset_high_recall",
        focal_alpha=0.75,  # High recall
        focal_gamma=2.0,
    )

    return results


if __name__ == "__main__":
    main()
