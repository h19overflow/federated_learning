"""
Centralized training orchestrator for pneumonia detection system.
Orchestrates complete training workflow from zip file or directory to trained model with comprehensive logging.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
from federated_pneumonia_detection.src.control.dl_model.utils.model.training_callbacks import (
    prepare_trainer_and_callbacks_pl,
    create_trainer_from_config,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet import (
    LitResNet,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.xray_data_module import (
    XRayDataModule,
)
from federated_pneumonia_detection.src.utils.data_processing import (
    load_metadata,
    create_train_val_split,
    sample_dataframe,
)
from federated_pneumonia_detection.models.system_constants import SystemConstants
from .utils import DataSourceExtractor


class CentralizedTrainer:
    """
    Centralized training orchestrator that handles complete training workflow.
    Accepts zip files or directories containing dataset and orchestrates all training components.
    """

    def __init__(
            self,
            config_path: Optional[str] = None,
            checkpoint_dir: str = "results/checkpoints",
            logs_dir: str = "results/training_logs",
    ):
        """
        Initialize centralized trainer.

        Args:
            config_path: Optional path to configuration file
            checkpoint_dir: Directory to save model checkpoints
            logs_dir: Directory to save training logs
        """
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.logger = self._setup_logging()

        # Load configuration
        config_loader = ConfigLoader()
        try:
            if config_path:
                # Load config as dictionary first
                self.logger.info(f"Loading configuration from: {config_path}")
                config_dict = config_loader.load_config(config_path)
                self.constants = config_loader.create_system_constants(config_dict)
                self.config = config_loader.create_experiment_config(config_dict)

                # Log key configuration values
                self.logger.info(f"Configuration loaded - Epochs: {self.config.epochs}, "
                                 f"Batch Size: {self.config.batch_size}, "
                                 f"Learning Rate: {self.config.learning_rate}, "
                                 f"Weight Decay: {self.config.weight_decay}, "
                                 f"Fine-tune Layers: {self.config.fine_tune_layers_count}")
            else:
                self.logger.info("No config path provided, using defaults")
                self.constants = config_loader.create_system_constants()
                self.config = config_loader.create_experiment_config()
        except Exception as e:
            self.logger.warning(f"Configuration loading failed: {e}. Using defaults.")
            # Fallback to default configuration
            self.constants = config_loader.create_system_constants()
            self.config = config_loader.create_experiment_config()

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize utilities with error handling
        try:
            self.data_source_extractor = DataSourceExtractor(self.logger)
        except Exception as e:
            self.logger.error(f"Failed to initialize utilities: {e}")
            raise

    def train(
            self,
            source_path: str,
            experiment_name: str = "pneumonia_detection",
            csv_filename: Optional[str] = None,
            run_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Complete training workflow from zip file or directory.

        Args:
            source_path: Path to zip file or directory containing dataset
            experiment_name: Name for this training experiment
            csv_filename: Optional specific CSV filename to look for

        Returns:
            Dictionary with training results and paths
        """

        try:
            try:
                image_dir, csv_path = self.data_source_extractor.extract_and_validate(
                    source_path, csv_filename
                )
            except Exception as e:
                self.logger.error(f"  Error message: {str(e)}")
                raise

            try:
                train_df, val_df = self._prepare_dataset(csv_path, image_dir)
            except Exception as e:
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Step 3: Create data module
            try:
                data_module = self._create_data_module(train_df, val_df, image_dir)
            except Exception as e:
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Step 4: Build model and callbacks
            try:
                model, callbacks, metrics_collector = self._build_model_and_callbacks(
                    train_df, experiment_name, run_id
                )
            except Exception as e:
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Step 5: Create trainer
            try:
                trainer = self._build_trainer(callbacks, experiment_name)
            except Exception as e:
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Step 6: Train model
            try:
                trainer.fit(model, data_module)
            except Exception as e:
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Collect results
            self.logger.info("Collecting training results...")
            try:
                results = self._collect_training_results(
                    trainer, model, metrics_collector
                )
            except Exception as e:
                self.logger.error(f"  Error type: {type(e).__name__}")
                raise
            return results

        except Exception as e:
            self.logger.error(f"  Error message: {str(e)}")
            raise

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and configuration."""
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "logs_dir": self.logs_dir,
            "config": {
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "validation_split": self.config.validation_split,
            },
            "temp_dir_active": self.data_source_extractor.temp_extract_dir is not None,
        }

    def _build_model_and_callbacks(
            self,
            train_df: pd.DataFrame,
            experiment_name: str = "pneumonia_detection",
            run_id: Optional[int] = None,
    ) -> Tuple[LitResNet, list, Any]:
        """
        Build model and training callbacks.

        Args:
            train_df: Training dataframe for computing class weights
            experiment_name: Name for the experiment

        Returns:
            Tuple of (model, callbacks, metrics_collector)
        """
        self.logger.info("Setting up model and callbacks...")
        try:
            callback_config = prepare_trainer_and_callbacks_pl(
                train_df_for_weights=train_df,
                class_column=self.constants.TARGET_COLUMN,
                checkpoint_dir=self.checkpoint_dir,
                model_filename="pneumonia_model",
                constants=self.constants,
                config=self.config,
                metrics_dir=os.path.join(self.logs_dir, "metrics"),
                experiment_name=experiment_name,
                run_id=run_id,
                enable_db_persistence=True,
            )
        except Exception as e:
            self.logger.error(f"Failed to build model and callbacks: {e}")
        try:
            model = LitResNet(
                constants=self.constants,
                config=self.config,
                class_weights_tensor=callback_config["class_weights"],
                monitor_metric="val_recall",
            )
        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")

        return model, callback_config["callbacks"], callback_config["metrics_collector"]

    def _build_trainer(self, callbacks: list, experiment_name: str) -> pl.Trainer:
        """
        Build PyTorch Lightning trainer.

        Args:
            callbacks: List of callbacks to use
            experiment_name: Name for experiment logging

        Returns:
            Configured trainer instance
        """
        self.logger.info("Setting up trainer...")
        try:
            tb_logger = TensorBoardLogger(
                save_dir=self.logs_dir, name=experiment_name, version=None
            )
        except Exception as e:
            self.logger.error(f"Failed to create TensorBoard logger: {e}")
            tb_logger = None
        try:
            trainer = create_trainer_from_config(
                constants=self.constants,
                config=self.config,
                callbacks=callbacks,
            )
        except Exception as e:
            self.logger.error(f"Failed to create trainer: {e}")
            trainer = None
        if tb_logger:
            trainer.logger = tb_logger

        return trainer

    def _collect_training_results(
            self, trainer: pl.Trainer, model: LitResNet, metrics_collector: Any
    ) -> Dict[str, Any]:
        """
        Collect and organize training results.

        Args:
            trainer: Trained PyTorch Lightning trainer
            model: Trained model instance
            metrics_collector: MetricsCollectorCallback instance

        Returns:
            Dictionary with training results
        """
        # Extract ModelCheckpoint callback from trainer
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                checkpoint_callback = callback
                break

        best_model_path = None
        best_model_score = None

        if checkpoint_callback:
            best_model_path = (
                checkpoint_callback.best_model_path
                if checkpoint_callback.best_model_path
                else None
            )
            if checkpoint_callback.best_model_score is not None:
                best_model_score = (
                    checkpoint_callback.best_model_score.item()
                    if hasattr(checkpoint_callback.best_model_score, "item")
                    else float(checkpoint_callback.best_model_score)
                )

        # Safely extract trainer state
        state_value = None
        if trainer.state and hasattr(trainer.state, "stage") and trainer.state.stage:
            state_value = (
                trainer.state.stage.value
                if hasattr(trainer.state.stage, "value")
                else str(trainer.state.stage)
            )

        # Extract metrics history from collector
        metrics_history = (
            metrics_collector.get_metrics_history() if metrics_collector else []
        )
        metrics_metadata = metrics_collector.get_metadata() if metrics_collector else {}

        results = {
            "best_model_path": best_model_path,
            "best_model_score": best_model_score,
            "current_epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "state": state_value,
            "model_summary": model.get_model_summary(),
            "checkpoint_dir": self.checkpoint_dir,
            "logs_dir": self.logs_dir,
            "metrics_history": metrics_history,
            "metrics_metadata": metrics_metadata,
            "total_epochs_trained": len(metrics_history),
        }

        self.logger.info(
            f"Training results collected: {len(metrics_history)} epochs tracked"
        )

        return results

    def _prepare_dataset(
            self, csv_path: str, image_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and validation datasets.

        Args:
            csv_path: Path to metadata CSV file
            image_dir: Directory containing images

        Returns:
            Tuple of (train_df, val_df)
        """

        # Create temporary constants with extracted paths
        temp_constants = SystemConstants(
            BASE_PATH=str(Path(csv_path).parent),
            METADATA_FILENAME=Path(csv_path).name,
            MAIN_IMAGES_FOLDER="",
            IMAGES_SUBFOLDER="",
            PATIENT_ID_COLUMN=self.constants.PATIENT_ID_COLUMN,
            TARGET_COLUMN=self.constants.TARGET_COLUMN,
            FILENAME_COLUMN=self.constants.FILENAME_COLUMN,
            IMAGE_EXTENSION=self.constants.IMAGE_EXTENSION,
            SEED=self.constants.SEED,
        )

        # Load metadata
        df = load_metadata(csv_path, temp_constants, self.logger)
        self.logger.info(f"Loaded metadata: {len(df)} samples from {csv_path}")

        # Sample data if needed
        if self.config.sample_fraction < 1.0:
            df = sample_dataframe(
                df,
                self.config.sample_fraction,
                temp_constants.TARGET_COLUMN,
                self.config.seed,
                self.logger,
            )

        # Create train/val split
        train_df, val_df = create_train_val_split(
            df,
            self.config.validation_split,
            temp_constants.TARGET_COLUMN,
            self.config.seed,
            self.logger,
        )

        self.logger.info(
            f"Dataset prepared: {len(train_df)} train, {len(val_df)} validation"
        )
        return train_df, val_df

    def _create_data_module(
            self, train_df: pd.DataFrame, val_df: pd.DataFrame, image_dir: str
    ) -> XRayDataModule:
        """
        Create PyTorch Lightning DataModule.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            image_dir: Directory containing images

        Returns:
            XRayDataModule instance
        """
        self.logger.info("Setting up data module...")

        data_module = XRayDataModule(
            train_df=train_df,
            val_df=val_df,
            constants=self.constants,
            config=self.config,
            image_dir=image_dir,
            validate_images_on_init=False,
        )

        return data_module

    # HELPER FUNCTIONS
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger(__name__)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger
