"""
Centralized training orchestrator for pneumonia detection system.
Orchestrates complete training workflow from zip file or directory to trained model with comprehensive logging.  # noqa: E501
"""

import os
from typing import Any, Dict, Optional

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.internals.loggers.logger import get_logger

from .centralized_trainer_utils import (
    build_model_and_callbacks,
    build_trainer,
    collect_training_results,
    complete_training_run,
    create_data_module,
    create_training_run,
    fail_training_run,
    prepare_dataset,
)
from .internals import DataSourceExtractor


class CentralizedTrainer:
    """
    Centralized training orchestrator that handles complete training workflow.
    Accepts zip files or directories containing dataset and orchestrates all training components.  # noqa: E501
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
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.data_source_extractor = DataSourceExtractor(self.logger)

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
        if run_id is None:
            run_id = create_training_run(source_path, experiment_name, self.logger)
        else:
            self.logger.info(f"Using existing run_id={run_id}")

        try:
            image_dir, csv_path = self.data_source_extractor.extract_and_validate(
                source_path,
                csv_filename,
            )
            train_df, val_df = prepare_dataset(
                csv_path,
                image_dir,
                self.config,
                self.logger,
            )
            data_module = create_data_module(
                train_df,
                val_df,
                image_dir,
                self.config,
                self.logger,
            )
            model, callbacks, metrics_collector = build_model_and_callbacks(
                train_df,
                self.config,
                self.checkpoint_dir,
                self.logs_dir,
                self.logger,
                experiment_name,
                run_id,
            )
            trainer = build_trainer(
                self.config,
                callbacks,
                self.logs_dir,
                experiment_name,
                self.logger,
            )
            trainer.fit(model, data_module)

            if run_id:
                complete_training_run(run_id, self.logger)

            results = collect_training_results(
                trainer,
                model,
                metrics_collector,
                self.logs_dir,
                self.checkpoint_dir,
                self.logger,
                run_id,
            )
            return results

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            if run_id:
                fail_training_run(run_id, self.logger)
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

    def _load_config(self, config_path: Optional[str]) -> ConfigManager:
        """Load configuration from path or use defaults."""
        try:
            config = ConfigManager(config_path) if config_path else ConfigManager()
            self.logger.info(
                f"Config loaded - Epochs: {config.get('experiment.epochs')}, "
                f"Batch: {config.get('experiment.batch_size')}, "
                f"LR: {config.get('experiment.learning_rate')}",
            )
            return config
        except Exception as e:
            self.logger.warning(f"Config load failed: {e}. Using defaults.")
            return ConfigManager()
