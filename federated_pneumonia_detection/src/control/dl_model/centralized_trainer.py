"""
Centralized training orchestrator for pneumonia detection system.
Orchestrates complete training workflow from zip file or directory to trained model with comprehensive logging.
"""

import os
import logging
from typing import Optional, Dict, Any

from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
from federated_pneumonia_detection.src.boundary.CRUD.experiment import experiment_crud
from federated_pneumonia_detection.src.boundary.engine import get_session
from .utils import DataSourceExtractor, DatasetPreparer, TrainerBuilder


class CentralizedTrainer:
    """
    Centralized training orchestrator that handles complete training workflow.
    Accepts zip files or directories containing dataset and orchestrates all training components.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_dir: str = "results/checkpoints",
        logs_dir: str = "results/training_logs"
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
            self.data_source_extractor = DataSourceExtractor(self.logger)
            self.dataset_preparer = DatasetPreparer(self.constants, self.config)
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
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING TRAINING WORKFLOW")
        self.logger.info("=" * 80)
        self.logger.info(f"Source path: {source_path}")
        self.logger.info(f"Experiment name: {experiment_name}")
        self.logger.info(f"CSV filename: {csv_filename if csv_filename else 'auto-detect'}")
        self.logger.info(f"Source type: {'ZIP file' if source_path.endswith('.zip') else 'Directory'}")

        try:
            # Step 1: Extract and validate data source
            self.logger.info("-" * 80)
            self.logger.info("STEP 1/6: Extracting and validating data source...")
            self.logger.info("-" * 80)
            try:
                image_dir, csv_path = self.data_source_extractor.extract_and_validate(source_path, csv_filename)
                self.logger.info(f"✓ Data source validated successfully")
                self.logger.info(f"  - Image directory: {image_dir}")
                self.logger.info(f"  - CSV file: {csv_path}")
            except Exception as e:
                self.logger.error(f"✗ Failed to extract/validate data source")
                self.logger.error(f"  Error type: {type(e).__name__}")
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Step 2: Load and process dataset
            self.logger.info("-" * 80)
            self.logger.info("STEP 2/6: Loading and processing dataset...")
            self.logger.info("-" * 80)
            try:
                train_df, val_df = self.dataset_preparer.prepare_dataset(csv_path, image_dir)
                self.logger.info(f"✓ Dataset prepared successfully")
                self.logger.info(f"  - Training samples: {len(train_df)}")
                self.logger.info(f"  - Validation samples: {len(val_df)}")
            except Exception as e:
                self.logger.error(f"✗ Failed to prepare dataset")
                self.logger.error(f"  Error type: {type(e).__name__}")
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Step 3: Create data module
            self.logger.info("-" * 80)
            self.logger.info("STEP 3/6: Creating data module...")
            self.logger.info("-" * 80)
            try:
                data_module = self.dataset_preparer.create_data_module(train_df, val_df, image_dir)
                self.logger.info(f"✓ Data module created successfully")
            except Exception as e:
                self.logger.error(f"✗ Failed to create data module")
                self.logger.error(f"  Error type: {type(e).__name__}")
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Step 4: Build model and callbacks
            self.logger.info("-" * 80)
            self.logger.info("STEP 4/6: Building model and callbacks...")
            self.logger.info("-" * 80)
            try:
                model, callbacks = self.trainer_builder.build_model_and_callbacks(train_df)
                self.logger.info(f"✓ Model and callbacks built successfully")
                self.logger.info(f"  - Callbacks: {len(callbacks)}")
            except Exception as e:
                self.logger.error(f"✗ Failed to build model and callbacks")
                self.logger.error(f"  Error type: {type(e).__name__}")
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Step 5: Create trainer
            self.logger.info("-" * 80)
            self.logger.info("STEP 5/6: Creating PyTorch Lightning trainer...")
            self.logger.info("-" * 80)
            try:
                trainer = self.trainer_builder.build_trainer(callbacks, experiment_name)
                self.logger.info(f"✓ Trainer created successfully")
                self.logger.info(f"  - Epochs: {self.config.epochs}")
                self.logger.info(f"  - Checkpoint dir: {self.checkpoint_dir}")
            except Exception as e:
                self.logger.error(f"✗ Failed to create trainer")
                self.logger.error(f"  Error type: {type(e).__name__}")
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Step 6: Train model
            self.logger.info("-" * 80)
            self.logger.info("STEP 6/6: Training model...")
            self.logger.info("-" * 80)
            try:
                trainer.fit(model, data_module)
                self.logger.info(f"✓ Model training completed")
            except Exception as e:
                self.logger.error(f"✗ Training failed during model.fit()")
                self.logger.error(f"  Error type: {type(e).__name__}")
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Collect results
            self.logger.info("-" * 80)
            self.logger.info("Collecting training results...")
            self.logger.info("-" * 80)
            try:
                results = self.trainer_builder.collect_training_results(trainer, model)
                self.logger.info(f"✓ Results collected successfully")
            except Exception as e:
                self.logger.error(f"✗ Failed to collect training results")
                self.logger.error(f"  Error type: {type(e).__name__}")
                self.logger.error(f"  Error message: {str(e)}")
                raise

            # Save results to database
            self.logger.info("-" * 80)
            self.logger.info("Saving results to database...")
            self.logger.info("-" * 80)
            try:
                db = get_session()
                # Create experiment if not exists
                exp = experiment_crud.get_by_name(db, experiment_name)
                if not exp:
                    exp = experiment_crud.create(db, name=experiment_name, description=f"Centralized training run for {experiment_name}")
                    db.commit()
                    self.logger.info(f"Created new experiment: {exp.id}")
                
                run_id = self.trainer_builder.save_results_to_db(
                    trainer, 
                    model,
                    experiment_id=exp.id,
                    training_mode="centralized",
                    source_path=source_path
                )
                results['run_id'] = run_id
                self.logger.info(f"✓ Results saved to database successfully")
                db.close()
            except Exception as e:
                self.logger.error(f"✗ Failed to save results to database")
                self.logger.error(f"  Error type: {type(e).__name__}")
                self.logger.error(f"  Error message: {str(e)}")
                self.logger.warning("Training completed but database save failed - continuing without database persistence")

            self.logger.info("=" * 80)
            self.logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            return results

        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error("TRAINING FAILED!")
            self.logger.error("=" * 80)
            self.logger.error(f"Final error: {type(e).__name__}: {str(e)}")

            # Log detailed traceback
            import traceback
            self.logger.error("Full traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self.logger.error(f"  {line}")
            raise
        finally:
            self.logger.info("-" * 80)
            self.logger.info("Cleaning up temporary files...")
            self.data_source_extractor.cleanup()
            self.logger.info("Cleanup completed")

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
        return self.data_source_extractor.validate_contents(source_path)

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
                'epochs': self.config.epochs,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'validation_split': self.config.validation_split
            },
            'temp_dir_active': self.data_source_extractor.temp_extract_dir is not None
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