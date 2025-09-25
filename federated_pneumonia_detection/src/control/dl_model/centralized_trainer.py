"""
Centralized training orchestrator for pneumonia detection system.
Orchestrates complete training workflow from zip file to trained model with comprehensive logging.
"""

import os
import logging
import zipfile
import tempfile
import shutil
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from federated_pneumonia_detection.src.entities.system_constants import SystemConstants
from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
from federated_pneumonia_detection.src.utils.data_processing import load_metadata, create_train_val_split, sample_dataframe
from federated_pneumonia_detection.src.control.dl_model.xray_data_module import XRayDataModule
from .lit_resnet import LitResNet
from federated_pneumonia_detection.src.control.dl_model.training_callbacks import prepare_trainer_and_callbacks_pl, create_trainer_from_config


class CentralizedTrainer:
    """
    Centralized training orchestrator that handles complete training workflow.
    Accepts zip files containing dataset and orchestrates all training components.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        logs_dir: str = "training_logs"
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
        self.temp_extract_dir = None

        # Setup logging
        self.logger = self._setup_logging()

        # Load configuration
        config_loader = ConfigLoader()
        if config_path:
            self.constants = config_loader.create_system_constants(config_path)
            self.config = config_loader.create_experiment_config(config_path)
        else:
            self.constants = config_loader.create_system_constants()
            self.config = config_loader.create_experiment_config()

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.logger.info(f"CentralizedTrainer initialized")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.info(f"Logs directory: {self.logs_dir}")

    def train_from_zip(
        self,
        zip_path: str,
        experiment_name: str = "pneumonia_detection",
        csv_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete training workflow from zip file.

        Args:
            zip_path: Path to zip file containing dataset
            experiment_name: Name for this training experiment
            csv_filename: Optional specific CSV filename to look for

        Returns:
            Dictionary with training results and paths
        """
        self.logger.info(f"Starting training from zip: {zip_path}")

        try:
            # Extract and validate zip file
            image_dir, csv_path = self._extract_and_validate_zip(zip_path, csv_filename)

            # Load and process data
            train_df, val_df = self._prepare_dataset(csv_path, image_dir)

            # Setup data module
            data_module = self._setup_data_module(train_df, val_df, image_dir)

            # Setup model and callbacks
            model, callbacks = self._setup_model_and_callbacks(train_df)

            # Create trainer
            trainer = self._setup_trainer(callbacks, experiment_name)

            # Train model
            self.logger.info("Starting training...")
            trainer.fit(model, data_module)

            # Get training results
            results = self._collect_training_results(trainer, model)

            self.logger.info("Training completed successfully!")
            return results

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            # Cleanup temporary directory
            self._cleanup_temp_directory()

    def _extract_and_validate_zip(
        self,
        zip_path: str,
        csv_filename: Optional[str] = None
    ) -> Tuple[str, str]:
        """Extract zip file and validate contents."""
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        self.temp_extract_dir = tempfile.mkdtemp(prefix="pneumonia_training_")
        self.logger.info(f"Extracting to temporary directory: {self.temp_extract_dir}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_extract_dir)

            # Find CSV file
            csv_files = list(Path(self.temp_extract_dir).rglob("*.csv"))
            if not csv_files:
                raise ValueError("No CSV files found in zip")

            if csv_filename:
                csv_path = None
                for csv_file in csv_files:
                    if csv_file.name == csv_filename:
                        csv_path = str(csv_file)
                        break
                if not csv_path:
                    raise ValueError(f"Specified CSV file not found: {csv_filename}")
            else:
                csv_path = str(csv_files[0])

            self.logger.info(f"Found CSV file: {csv_path}")

            # Find image directory (look for common image extensions)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(Path(self.temp_extract_dir).rglob(f"*{ext}")))

            if not image_files:
                raise ValueError("No image files found in zip")

            # Determine image directory (parent of first image)
            image_dir = str(image_files[0].parent)
            self.logger.info(f"Found image directory: {image_dir}")
            self.logger.info(f"Total images found: {len(image_files)}")

            return image_dir, csv_path

        except Exception as e:
            self.logger.error(f"Failed to extract zip file: {e}")
            raise

    def _prepare_dataset(self, csv_path: str, image_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare training and validation datasets."""
        self.logger.info("Loading and processing dataset...")

        # Update constants with extracted paths
        temp_constants = SystemConstants(
            BASE_PATH=str(Path(csv_path).parent),
            METADATA_FILENAME=Path(csv_path).name,
            MAIN_IMAGES_FOLDER="",  # Already pointing to image dir
            IMAGES_SUBFOLDER="",
            PATIENT_ID_COLUMN=self.constants.PATIENT_ID_COLUMN,
            TARGET_COLUMN=self.constants.TARGET_COLUMN,
            FILENAME_COLUMN=self.constants.FILENAME_COLUMN,
            IMAGE_EXTENSION=self.constants.IMAGE_EXTENSION,
            random_seed=self.constants.random_seed
        )

        # Load metadata
        df = load_metadata(csv_path, temp_constants, self.logger)
        self.logger.info(f"Loaded metadata: {len(df)} samples")

        # Sample data if needed
        if self.config.sample_fraction < 1.0:
            df = sample_dataframe(
                df,
                self.config.sample_fraction,
                temp_constants.TARGET_COLUMN,
                self.config.seed,
                self.logger
            )

        # Create train/val split
        train_df, val_df = create_train_val_split(
            df,
            self.config.validation_split,
            temp_constants.TARGET_COLUMN,
            self.config.seed,
            self.logger
        )

        self.logger.info(f"Dataset prepared: {len(train_df)} train, {len(val_df)} validation")
        return train_df, val_df

    def _setup_data_module(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        image_dir: str
    ) -> XRayDataModule:
        """Setup PyTorch Lightning DataModule."""
        self.logger.info("Setting up data module...")

        data_module = XRayDataModule(
            train_df=train_df,
            val_df=val_df,
            constants=self.constants,
            config=self.config,
            image_dir=image_dir,
            validate_images_on_init=False  # Skip validation for performance
        )

        return data_module

    def _setup_model_and_callbacks(
        self,
        train_df: pd.DataFrame
    ) -> Tuple[LitResNet, list]:
        """Setup model and training callbacks."""
        self.logger.info("Setting up model and callbacks...")

        # Prepare callbacks and get class weights
        callback_config = prepare_trainer_and_callbacks_pl(
            train_df_for_weights=train_df,
            class_column=self.constants.TARGET_COLUMN,
            checkpoint_dir=self.checkpoint_dir,
            model_filename="pneumonia_model",
            constants=self.constants,
            config=self.config
        )

        # Create model
        model = LitResNet(
            constants=self.constants,
            config=self.config,
            class_weights_tensor=callback_config['class_weights'],
            monitor_metric="val_recall"
        )

        self.logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

        return model, callback_config['callbacks']

    def _setup_trainer(self, callbacks: list, experiment_name: str) -> pl.Trainer:
        """Setup PyTorch Lightning trainer."""
        self.logger.info("Setting up trainer...")

        # Setup logger
        tb_logger = TensorBoardLogger(
            save_dir=self.logs_dir,
            name=experiment_name,
            version=None  # Auto-generate version
        )

        trainer = create_trainer_from_config(
            constants=self.constants,
            config=self.config,
            callbacks=callbacks
        )

        # Update trainer with logger
        trainer.logger = tb_logger

        return trainer

    def _collect_training_results(
        self,
        trainer: pl.Trainer,
        model: LitResNet
    ) -> Dict[str, Any]:
        """Collect and organize training results."""
        results = {
            'best_model_path': trainer.checkpoint_callback.best_model_path,
            'best_model_score': trainer.checkpoint_callback.best_model_score.item(),
            'current_epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'model_summary': model.get_model_summary(),
            'checkpoint_dir': self.checkpoint_dir,
            'logs_dir': self.logs_dir
        }

        # Add callback metrics if available
        if trainer.callback_metrics:
            results['final_metrics'] = {
                k: v.item() if hasattr(v, 'item') else v
                for k, v in trainer.callback_metrics.items()
            }

        self.logger.info(f"Training results collected: {results}")
        return results

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger(__name__)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    def _cleanup_temp_directory(self):
        """Clean up temporary extraction directory."""
        if self.temp_extract_dir and os.path.exists(self.temp_extract_dir):
            try:
                shutil.rmtree(self.temp_extract_dir)
                self.logger.info("Temporary directory cleaned up")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp directory: {e}")

    # HELPER FUNCTIONS
    def validate_zip_contents(self, zip_path: str) -> Dict[str, Any]:
        """
        Validate zip file contents without extraction.

        Args:
            zip_path: Path to zip file

        Returns:
            Dictionary with validation results
        """
        if not os.path.exists(zip_path):
            return {'valid': False, 'error': 'Zip file not found'}

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()

            csv_files = [f for f in file_list if f.endswith('.csv')]
            image_files = [f for f in file_list if any(f.lower().endswith(ext)
                          for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])]

            validation = {
                'valid': len(csv_files) > 0 and len(image_files) > 0,
                'csv_files': csv_files,
                'image_count': len(image_files),
                'total_files': len(file_list),
                'error': None
            }

            if not validation['valid']:
                validation['error'] = f"Missing required files. CSV: {len(csv_files)}, Images: {len(image_files)}"

            return validation

        except Exception as e:
            return {'valid': False, 'error': f'Failed to read zip: {e}'}

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and configuration."""
        return {
            'checkpoint_dir': self.checkpoint_dir,
            'logs_dir': self.logs_dir,
            'config': {
                'max_epochs': self.config.max_epochs,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'validation_split': self.config.validation_split
            },
            'temp_dir_active': self.temp_extract_dir is not None
        }