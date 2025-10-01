"""
Centralized training orchestrator for pneumonia detection system.
Orchestrates complete training workflow from zip file or directory to trained model with comprehensive logging.
"""

import os
import logging
from typing import Optional, Dict, Any

from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
from .utils import ZipHandler, DirectoryHandler, DatasetPreparer, TrainerBuilder


class CentralizedTrainer:
    """
    Centralized training orchestrator that handles complete training workflow.
    Accepts zip files or directories containing dataset and orchestrates all training components.
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
        self.logger = self._setup_logging()

        # Load configuration
        config_loader = ConfigLoader()
        try:
            if config_path:
                # Load config as dictionary first
                config_dict = config_loader.load_config(config_path)
                self.constants = config_loader.create_system_constants(config_dict)
                self.config = config_loader.create_experiment_config(config_dict)
            else:
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
            self.zip_handler = ZipHandler(self.logger)
            self.directory_handler = DirectoryHandler(self.logger)
            self.dataset_preparer = DatasetPreparer(self.constants, self.config, self.logger)
            self.trainer_builder = TrainerBuilder(
                self.constants, self.config, self.checkpoint_dir, self.logs_dir, self.logger
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize utilities: {e}")
            raise

        self.logger.info(f"CentralizedTrainer initialized")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.info(f"Logs directory: {self.logs_dir}")

    def train(
        self,
        source_path: str,
        experiment_name: str = "pneumonia_detection",
        csv_filename: Optional[str] = None
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
        self.logger.info(f"Starting training from: {source_path}")

        try:
            # Detect source type and extract/find data
            if os.path.isfile(source_path):
                self.logger.info("Detected zip file")
                image_dir, csv_path = self.zip_handler.extract_and_validate(source_path, csv_filename)
            elif os.path.isdir(source_path):
                self.logger.info("Detected directory")
                image_dir, csv_path = self.directory_handler.find_data(source_path, csv_filename)
            else:
                raise ValueError(f"Invalid source path: {source_path}. Must be a zip file or directory.")

            # Load and process data
            train_df, val_df = self.dataset_preparer.prepare_dataset(csv_path, image_dir)

            # Setup data module
            data_module = self.dataset_preparer.create_data_module(train_df, val_df, image_dir)

            # Setup model and callbacks
            model, callbacks = self.trainer_builder.build_model_and_callbacks(train_df)

            # Create trainer
            trainer = self.trainer_builder.build_trainer(callbacks, experiment_name)

            # Train model
            self.logger.info("Starting training...")
            trainer.fit(model, data_module)

            # Get training results
            results = self.trainer_builder.collect_training_results(trainer, model)

            self.logger.info("Training completed successfully!")
            return results

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self.zip_handler.cleanup()

    def train_from_zip(
        self,
        zip_path: str,
        experiment_name: str = "pneumonia_detection",
        csv_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Backward compatibility wrapper for train().

        Args:
            zip_path: Path to zip file containing dataset
            experiment_name: Name for this training experiment
            csv_filename: Optional specific CSV filename to look for

        Returns:
            Dictionary with training results and paths
        """
        return self.train(zip_path, experiment_name, csv_filename)

    def validate_source(self, source_path: str) -> Dict[str, Any]:
        """
        Validate source contents without processing.

        Args:
            source_path: Path to zip file or directory

        Returns:
            Dictionary with validation results
        """
        try:
            if os.path.isfile(source_path):
                result = self.zip_handler.validate_contents(source_path)
            elif os.path.isdir(source_path):
                result = self.directory_handler.validate_contents(source_path)
            else:
                return {'valid': False, 'error': 'Path is neither a file nor a directory'}

            # Normalize result to dictionary format
            if isinstance(result, dict):
                return result
            elif isinstance(result, str):
                # If handler returns a string, assume it's valid if non-empty
                return {'valid': bool(result.strip()), 'error': result if not result.strip() else None}
            elif hasattr(result, '__bool__'):
                # If it has a boolean value
                return {'valid': bool(result), 'error': None}
            else:
                # Default to valid for unknown types
                return {'valid': True, 'error': None}

        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}

    def validate_zip_contents(self, zip_path: str) -> Dict[str, Any]:
        """
        Backward compatibility wrapper for validate_source().

        Args:
            zip_path: Path to zip file

        Returns:
            Dictionary with validation results
        """
        return self.validate_source(zip_path)

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
            'temp_dir_active': self.zip_handler.temp_extract_dir is not None
        }

    # HELPER FUNCTIONS
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