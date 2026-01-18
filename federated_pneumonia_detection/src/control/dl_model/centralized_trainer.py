"""
Centralized training orchestrator for pneumonia detection system.
Orchestrates complete training workflow from zip file or directory to trained model with comprehensive logging.
"""

import os
import logging
import json
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING
from datetime import datetime

import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.control.dl_model.utils.model.training_callbacks import (
    prepare_trainer_and_callbacks_pl,
    create_trainer_from_config,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet import (
    LitResNet,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.lit_resnet_enhanced import (
    LitResNetEnhanced,
    ProgressiveUnfreezeCallback,
)
from federated_pneumonia_detection.src.control.dl_model.utils.model.xray_data_module import (
    XRayDataModule,
)
from federated_pneumonia_detection.src.utils.data_processing import (
    load_metadata,
    create_train_val_split,
)
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
        try:
            if config_path:
                self.logger.info(f"Loading configuration from: {config_path}")
                self.config = ConfigManager(config_path)
            else:
                self.logger.info("Using default configuration")
                self.config = ConfigManager()

            # Log key configuration values
            self.logger.info(
                f"Configuration loaded - Epochs: {self.config.get('experiment.epochs')}, "
                f"Batch Size: {self.config.get('experiment.batch_size')}, "
                f"Learning Rate: {self.config.get('experiment.learning_rate')}, "
                f"Weight Decay: {self.config.get('experiment.weight_decay')}, "
                f"Fine-tune Layers: {self.config.get('experiment.fine_tune_layers_count')}"
            )
        except Exception as e:
            self.logger.warning(f"Configuration loading failed: {e}. Using defaults.")
            # Fallback to default configuration
            self.config = ConfigManager()

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
            run_id: Optional pre-existing run ID (if None, creates new run)

        Returns:
            Dictionary with training results and paths
        """
        # Create run in database to track timing
        if run_id is None:
            self.logger.info("Creating centralized training run in database...")
            db = get_session()
            try:
                run_data = {
                    "training_mode": "centralized",
                    "status": "in_progress",
                    "start_time": datetime.now(),
                    "wandb_id": f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "source_path": source_path,
                }
                new_run = run_crud.create(db, **run_data)
                db.commit()
                run_id = new_run.id
                self.logger.info(f"Created run in database with id={run_id}")
            except Exception as e:
                self.logger.error(f"Failed to create run: {e}")
                db.rollback()
                run_id = None
            finally:
                db.close()
        else:
            self.logger.info(f"Using existing run_id={run_id}")

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

            # Mark run as completed in database
            if run_id:
                self.logger.info("Marking run as completed in database...")
                db = get_session()
                try:
                    run_crud.complete_run(db, run_id=run_id, status="completed")
                    db.commit()
                    self.logger.info(f"Run {run_id} marked as completed")
                except Exception as e:
                    self.logger.error(f"Failed to complete run: {e}")
                    db.rollback()
                finally:
                    db.close()

            # Collect results
            self.logger.info("Collecting training results...")
            try:
                results = self._collect_training_results(
                    trainer, model, metrics_collector, run_id
                )
            except Exception as e:
                self.logger.error(f"  Error type: {type(e).__name__}")
                raise
            return results

        except Exception as e:
            self.logger.error(f"  Error message: {str(e)}")

            # Mark run as failed if it exists
            if run_id:
                db = get_session()
                try:
                    run_crud.complete_run(db, run_id=run_id, status="failed")
                    db.commit()
                    self.logger.info(f"Run {run_id} marked as failed")
                except Exception as db_error:
                    self.logger.error(f"Failed to mark run as failed: {db_error}")
                    db.rollback()
                finally:
                    db.close()

            raise

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and configuration."""
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "logs_dir": self.logs_dir,
            "config": {
                "epochs": self.config.get("experiment.epochs"),
                "learning_rate": self.config.get("experiment.learning_rate"),
                "batch_size": self.config.get("experiment.batch_size"),
                "validation_split": self.config.get("experiment.validation_split"),
            },
            "temp_dir_active": self.data_source_extractor.temp_extract_dir is not None,
        }

    def _build_model_and_callbacks(
        self,
        train_df: pd.DataFrame,
        experiment_name: str = "pneumonia_detection",
        run_id: Optional[int] = None,
        is_federated: bool = False,
        client_id: Optional[int] = None,
        round_number: int = 0,
    ) -> Tuple[pl.LightningModule, list, Any]:
        """
        Build model and training callbacks.

        Args:
            train_df: Training dataframe for computing class weights
            experiment_name: Name for the experiment
            run_id: Optional database run ID for metrics persistence
            is_federated: If True, uses local_epochs (max-epochs); if False, uses epochs
            client_id: Optional client ID for federated learning context
            round_number: Round number for federated learning

        Returns:
            Tuple of (model, callbacks, metrics_collector)
        """
        self.logger.info("Setting up model and callbacks (Enhanced v3 - Balanced)...")
        if is_federated and client_id is not None:
            self.logger.info(
                f"[CentralizedTrainer] Federated mode enabled for client_id={client_id}, round={round_number}"
            )

        # Get sampling intervals from config
        batch_interval = self.config.get("experiment.batch_sample_interval", 10)
        gradient_interval = self.config.get("experiment.gradient_sample_interval", 20)

        callback_config = prepare_trainer_and_callbacks_pl(
            train_df_for_weights=train_df,
            class_column=self.config.get("columns.target"),
            checkpoint_dir=self.checkpoint_dir,
            model_filename="pneumonia_model",
            config=self.config,
            metrics_dir=os.path.join(self.logs_dir, "metrics"),
            experiment_name=experiment_name,
            run_id=run_id,
            enable_db_persistence=True,
            is_federated=is_federated,
            client_id=client_id,
            round_number=round_number,
            batch_sample_interval=batch_interval,
            gradient_sample_interval=gradient_interval,
        )

        # Enhanced v3 - Balanced configuration for high accuracy, precision, recall, AND F1
        # focal_alpha=0.6 balances recall and precision
        # focal_gamma=1.5 provides moderate focus on hard examples
        # Keep val_recall as monitor for frontend compatibility
        model = LitResNetEnhanced(
            config=self.config,
            class_weights_tensor=callback_config["class_weights"],
            use_focal_loss=True,
            focal_alpha=0.6,  # Balanced between recall and precision
            focal_gamma=1.5,  # Less extreme focus
            label_smoothing=0.05,  # Mild regularization
            use_cosine_scheduler=True,
            monitor_metric="val_recall",  # Keep val_recall for frontend/callback compatibility
        )

        # Add progressive unfreezing callback for better feature learning
        total_epochs = self.config.get("experiment.epochs", 25)
        unfreeze_epochs = [
            int(total_epochs * 0.15),
            int(total_epochs * 0.35),
            int(total_epochs * 0.55),
            int(total_epochs * 0.75),
        ]
        progressive_callback = ProgressiveUnfreezeCallback(
            unfreeze_epochs=unfreeze_epochs,
            layers_per_unfreeze=4,
        )
        self.logger.info(f"Progressive unfreezing at epochs: {unfreeze_epochs}")

        callbacks = callback_config["callbacks"] + [progressive_callback]

        return model, callbacks, callback_config["metrics_collector"]

    def _build_trainer(
        self, callbacks: list, experiment_name: str, is_federated: bool = False
    ) -> pl.Trainer:
        """
        Build PyTorch Lightning trainer.

        Args:
            callbacks: List of callbacks to use
            experiment_name: Name for experiment logging
            is_federated: If True, uses max-epochs; if False, uses epochs

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
                config=self.config,
                callbacks=callbacks,
                is_federated=is_federated,
            )
        except Exception as e:
            self.logger.error(f"Failed to create trainer: {e}")
            trainer = None
        if tb_logger:
            trainer.logger = tb_logger

        return trainer

    def _collect_training_results(
        self, trainer: pl.Trainer, model: LitResNet, metrics_collector: Any, run_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Collect and organize training results.

        Args:
            trainer: Trained PyTorch Lightning trainer
            model: Trained model instance
            metrics_collector: MetricsCollectorCallback instance
            run_id: Optional database run ID

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
            "run_id": run_id,
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

        self.logger.info(
            f"Saving metrics to: results/centralized/metrics_output/metrics.json"
        )

        output_path = os.path.join(self.logs_dir, "metrics_output", "metrics.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f)

        return results

    def _prepare_dataset(
        self, csv_path: str, image_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare training dataset.

        Args:
            csv_path: Path to metadata CSV file
            image_dir: Directory containing images

        Returns:
            Tuple of (train_df, val_df)
        """

        # Load metadata using ConfigManager directly
        df = load_metadata(csv_path, self.config, self.logger)
        self.logger.info(f"Loaded metadata: {len(df)} samples from {csv_path}")

        # Create train/val split (use all uploaded data, no sampling)
        target_col = self.config.get("columns.target")
        train_df, val_df = create_train_val_split(
            df,
            self.config.get("experiment.validation_split"),
            target_col,
            self.config.get("experiment.seed"),
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
