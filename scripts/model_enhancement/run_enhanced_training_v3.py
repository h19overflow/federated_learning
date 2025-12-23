"""
Enhanced training v3 - Balanced high-accuracy configuration.

Targets: High accuracy, precision, recall, and F1 simultaneously.

Key changes from v2:
- focal_alpha=0.6 (balanced between recall and precision)
- Monitor val_f1 for balanced optimization
- Earlier progressive unfreezing for better feature learning
- Reduced focal_gamma=1.5 for less extreme focus
- Longer training (35 epochs) with more patience
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnhancedTrainerV3:
    """Balanced high-accuracy trainer targeting all metrics."""

    def __init__(self, checkpoint_dir: str, logs_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logs_dir = Path(logs_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.config = ConfigManager()
        self._apply_balanced_config()
        self.data_extractor = DataSourceExtractor(logger)

    def _apply_balanced_config(self):
        """Apply balanced configuration for all metrics."""
        defaults = {
            "system.sample_fraction": 1.0,
            "experiment.epochs": 35,
            "experiment.batch_size": 32,
            "experiment.learning_rate": 0.0005,  # Balanced LR
            "experiment.dropout_rate": 0.35,
            "experiment.early_stopping_patience": 12,
            "experiment.use_custom_preprocessing": True,
            "experiment.adaptive_histogram": True,
            "experiment.contrast_stretch": True,
            "experiment.fine_tune_layers_count": 0,
            "experiment.weight_decay": 0.0002,
        }
        for key, value in defaults.items():
            try:
                self.config.set(key, value)
            except Exception:
                pass
        logger.info("Balanced v3 configuration applied")

    def _compute_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        target_col = self.config.get("columns.target", "Target")
        y = np.array(train_df[target_col].values, dtype=int)
        unique_classes = np.unique(y)
        class_weights = compute_class_weight("balanced", classes=unique_classes, y=y)
        return torch.tensor(class_weights, dtype=torch.float32) if len(class_weights) == 2 else torch.tensor([1.0, 1.0], dtype=torch.float32)

    def train(
        self,
        source_path: str,
        experiment_name: str = "balanced_v3",
        # Balanced settings
        use_focal_loss: bool = True,
        focal_alpha: float = 0.6,  # Balanced
        focal_gamma: float = 1.5,  # Less extreme
        label_smoothing: float = 0.05,  # Mild smoothing
        use_cosine_scheduler: bool = True,
        use_progressive_unfreeze: bool = True,
    ) -> Dict[str, Any]:
        """Run balanced high-accuracy training."""
        logger.info("=" * 70)
        logger.info("ENHANCED TRAINING V3 - BALANCED HIGH ACCURACY")
        logger.info("=" * 70)
        logger.info(f"Target: High accuracy, precision, recall, AND F1")
        logger.info(f"Focal Alpha: {focal_alpha}, Gamma: {focal_gamma}")
        logger.info(f"Monitor: val_f1 (balanced metric)")

        # Extract data
        logger.info("\n[1/8] Extracting data...")
        image_dir, csv_path = self.data_extractor.extract_and_validate(source_path)

        # Prepare dataset
        logger.info("\n[2/8] Preparing dataset...")
        train_df, val_df = self._prepare_dataset(csv_path, image_dir)

        # Class distribution analysis
        target_col = self.config.get("columns.target", "Target")
        train_pos = (train_df[target_col] == 1).sum()
        train_neg = (train_df[target_col] == 0).sum()
        val_pos = (val_df[target_col] == 1).sum()
        val_neg = (val_df[target_col] == 0).sum()

        logger.info(f"Training: {len(train_df)} ({train_neg} neg, {train_pos} pos, {train_pos/len(train_df):.1%} positive)")
        logger.info(f"Validation: {len(val_df)} ({val_neg} neg, {val_pos} pos)")

        # Class weights
        logger.info("\n[3/8] Computing class weights...")
        class_weights = self._compute_class_weights(train_df)
        logger.info(f"Class weights: {class_weights}")

        # Data module
        logger.info("\n[4/8] Creating data module...")
        data_module = XRayDataModule(
            train_df=train_df,
            val_df=val_df,
            config=self.config,
            image_dir=image_dir,
            validate_images_on_init=False,
        )

        # Model - optimized for val_f1
        logger.info("\n[5/8] Creating balanced model...")
        model = LitResNetEnhanced(
            config=self.config,
            class_weights_tensor=class_weights,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            use_cosine_scheduler=use_cosine_scheduler,
            monitor_metric="val_f1",  # Monitor F1 for balance
        )
        logger.info(f"Parameters: {model.get_model_summary()['total_parameters']:,}")

        # Callbacks
        logger.info("\n[6/8] Setting up callbacks...")
        callbacks = self._create_callbacks(use_progressive_unfreeze)

        # Trainer
        logger.info("\n[7/8] Creating trainer...")
        trainer = self._create_trainer(callbacks, experiment_name)

        # Train
        logger.info("\n[8/8] Starting training...")
        logger.info(f"Epochs: {self.config.get('experiment.epochs')}, LR: {self.config.get('experiment.learning_rate')}")
        torch.set_float32_matmul_precision("high")
        trainer.fit(model, data_module)

        # Results
        results = self._collect_results(trainer, model)
        self._print_final_report(results)
        self._save_results(results, experiment_name)

        return results

    def _prepare_dataset(self, csv_path: str, image_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = load_metadata(csv_path, self.config, logger)
        target_col = self.config.get("columns.target", "Target")
        return create_train_val_split(
            df, self.config.get("experiment.validation_split", 0.2),
            target_col, self.config.get("experiment.seed", 42), logger
        )

    def _create_callbacks(self, use_progressive_unfreeze: bool) -> list:
        callbacks = []

        # Single main checkpoint by F1 (balanced metric)
        callbacks.append(ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename="v3_{epoch:02d}_{val_f1:.3f}_{val_recall:.3f}_{val_precision:.3f}",
            monitor="val_f1",
            mode="max",
            save_top_k=5,
            save_last=True,
            verbose=True,
        ))

        # Early stopping on F1
        callbacks.append(EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=self.config.get("experiment.early_stopping_patience", 12),
            min_delta=0.005,
            verbose=True,
        ))

        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        # Earlier progressive unfreezing
        if use_progressive_unfreeze:
            total_epochs = self.config.get("experiment.epochs", 35)
            unfreeze_epochs = [
                int(total_epochs * 0.15),  # Unfreeze earlier
                int(total_epochs * 0.35),
                int(total_epochs * 0.55),
                int(total_epochs * 0.75),
            ]
            callbacks.append(ProgressiveUnfreezeCallback(
                unfreeze_epochs=unfreeze_epochs,
                layers_per_unfreeze=4,  # More layers
            ))
            logger.info(f"Progressive unfreezing at: {unfreeze_epochs}")

        return callbacks

    def _create_trainer(self, callbacks: list, experiment_name: str) -> pl.Trainer:
        loggers = [
            TensorBoardLogger(str(self.logs_dir), name=experiment_name),
            CSVLogger(str(self.logs_dir), name=f"{experiment_name}_csv"),
        ]

        return pl.Trainer(
            max_epochs=self.config.get("experiment.epochs", 35),
            accelerator="auto",
            devices=1,
            precision="16-mixed",
            callbacks=callbacks,
            logger=loggers,
            enable_progress_bar=True,
            log_every_n_steps=1,
            gradient_clip_val=1.0,
            accumulate_grad_batches=1,
            deterministic=False,
        )

    def _collect_results(self, trainer: pl.Trainer, model: LitResNetEnhanced) -> Dict[str, Any]:
        best_f1_path = None
        best_f1_score = None
        best_recall_path = None
        best_recall_score = None

        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                if "val_f1" in str(callback.monitor) and callback.best_model_path:
                    if callback.best_model_score is not None:
                        score = float(callback.best_model_score)
                        if best_f1_score is None or score > best_f1_score:
                            best_f1_score = score
                            best_f1_path = callback.best_model_path
                if "val_recall" in str(callback.monitor) and callback.best_model_path:
                    if callback.best_model_score is not None:
                        score = float(callback.best_model_score)
                        if best_recall_score is None or score > best_recall_score:
                            best_recall_score = score
                            best_recall_path = callback.best_model_path

        return {
            "best_f1_model_path": best_f1_path,
            "best_f1_score": best_f1_score,
            "best_recall_model_path": best_recall_path,
            "best_recall_score": best_recall_score,
            "total_epochs": trainer.current_epoch,
            "configuration": {
                "epochs": self.config.get("experiment.epochs"),
                "learning_rate": self.config.get("experiment.learning_rate"),
                "focal_alpha": model.focal_alpha,
                "focal_gamma": model.focal_gamma,
                "label_smoothing": model.label_smoothing,
            }
        }

    def _print_final_report(self, results: Dict[str, Any]):
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING V3 COMPLETE - FINAL REPORT")
        logger.info("=" * 70)
        logger.info(f"\nBest F1 Score: {results.get('best_f1_score', 'N/A')}")
        logger.info(f"Best F1 Model: {results.get('best_f1_model_path', 'N/A')}")
        logger.info(f"\nBest Recall Score: {results.get('best_recall_score', 'N/A')}")
        logger.info(f"Best Recall Model: {results.get('best_recall_model_path', 'N/A')}")
        logger.info(f"\nTotal Epochs Trained: {results.get('total_epochs', 'N/A')}")

    def _save_results(self, results: Dict[str, Any], experiment_name: str):
        results_path = self.logs_dir / f"{experiment_name}_results.json"
        serializable = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None), dict)) else v for k, v in results.items()}
        if isinstance(results.get("configuration"), dict):
            serializable["configuration"] = results["configuration"]
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"\nResults saved to: {results_path}")


def main():
    source_path = r"C:\Users\User\Projects\FYP2\Training_Sample_5pct.zip"
    output_dir = project_root / "model_enhancement_results" / "balanced_v3"

    trainer = EnhancedTrainerV3(
        checkpoint_dir=str(output_dir / "checkpoints"),
        logs_dir=str(output_dir / "logs"),
    )

    results = trainer.train(
        source_path=source_path,
        experiment_name="balanced_pneumonia_v3",
        use_focal_loss=True,
        focal_alpha=0.6,  # Balanced
        focal_gamma=1.5,  # Less extreme
        label_smoothing=0.05,  # Mild smoothing
        use_cosine_scheduler=True,
        use_progressive_unfreeze=True,
    )

    return results


if __name__ == "__main__":
    main()
