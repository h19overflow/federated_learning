"""
Federated training orchestrator for pneumonia detection system.
Mirrors CentralizedTrainer API while orchestrating federated learning workflow.
"""

import os
import logging
from typing import Optional, Dict, Any

from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
from federated_pneumonia_detection.src.control.dl_model.utils.data import (
    DataSourceExtractor, DatasetPreparer
)
from federated_pneumonia_detection.src.control.federated_learning.data.partitioner import (
    partition_data_stratified
)
from federated_pneumonia_detection.src.control.federated_learning.core.simulation_runner import (
    SimulationRunner
)


class FederatedTrainer:
    """
    Federated training orchestrator that handles complete FL workflow.
    Mirrors CentralizedTrainer API for consistent interface.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_dir: str = "fed_results/federated_checkpoints",
        logs_dir: str = "fed_results/federated_logs",
        partition_strategy: str = "stratified"
    ):
        """
        Initialize federated trainer.

        Args:
            config_path: Optional path to configuration file
            checkpoint_dir: Directory to save model checkpoints
            logs_dir: Directory to save training logs
            partition_strategy: Data partitioning strategy ('iid', 'non-iid', 'stratified')
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.partition_strategy = partition_strategy
        self.logger = self._setup_logging()

        # Load configuration
        config_loader = ConfigLoader()
        try:
            if config_path:
                config_dict = config_loader.load_config(config_path)
                self.constants = config_loader.create_system_constants(config_dict)
                self.config = config_loader.create_experiment_config(config_dict)
            else:
                self.constants = config_loader.create_system_constants()
                self.config = config_loader.create_experiment_config()
        except Exception as e:
            self.logger.warning(f"Configuration loading failed: {e}. Using defaults.")
            self.constants = config_loader.create_system_constants()
            self.config = config_loader.create_experiment_config()

        # Update config checkpoint_dir if not set
        if not hasattr(self.config, 'checkpoint_dir'):
            self.config.checkpoint_dir = self.checkpoint_dir
        if not hasattr(self.config, 'logs_dir'):
            self.config.logs_dir = self.logs_dir

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        try:
            self.handler = DataSourceExtractor(self.logger)
            self.dataset_preparer = DatasetPreparer(self.constants, self.config)
        except Exception as e:
            self.logger.error(f"Failed to initialize utilities: {e}")
            raise

        # Initialize simulation runner
        self.simulation_runner = SimulationRunner(
            constants=self.constants,
            config=self.config,
            logger=self.logger
        )

        self.logger.info("FederatedTrainer initialized")
        self.logger.info(f"Partition strategy: {self.partition_strategy}")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.info(f"Logs directory: {self.logs_dir}")

    def train(
        self,
        source_path: str,
        experiment_name: str = "federated_pneumonia_detection",
        csv_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete federated training workflow from zip file or directory.

        Args:
            source_path: Path to zip file or directory containing dataset
            experiment_name: Name for this training experiment
            csv_filename: Optional specific CSV filename to look for

        Returns:
            Dictionary with training results and paths
        """
        self.logger.info(f"Starting federated training from: {source_path}")
        self.logger.info(f"Experiment name: {experiment_name}")

        try:
            # Detect source type and extract/find data (reuse from centralized)
            if os.path.isfile(source_path):
                self.logger.info("Detected zip file")
                image_dir, csv_path = self.handler.extract_and_validate(source_path, csv_filename)
            elif os.path.isdir(source_path):
                self.logger.info("Detected directory")
                image_dir, csv_path = self.handler.extract_and_validate(source_path, csv_filename)
            else:
                raise ValueError(f"Invalid source path: {source_path}")

            # Load and process data (reuse from centralized)
            train_df, val_df = self.dataset_preparer.prepare_dataset(csv_path, image_dir)

            # Combine train and val for federated partitioning
            import pandas as pd
            full_df = pd.concat([train_df, val_df], ignore_index=True)

            self.logger.info(f"Total samples for federated learning: {len(full_df)}")

            # Partition data across clients
            client_partitions = self._partition_data_for_clients(full_df)

            # Run federated learning simulation
            results = self._run_federated_simulation(
                client_partitions,
                image_dir,
                experiment_name
            )

            self.logger.info("Federated training completed successfully!")
            return results

        except Exception as e:
            self.logger.error(f"Federated training failed: {e}")
            raise
        finally:
            self.handler.cleanup()

    def _partition_data_for_clients(self, df) -> list:
        """
        Partition data across federated clients based on strategy.

        Args:
            df: Full dataset DataFrame

        Returns:
            List of DataFrames, one per client
        """
        num_clients = self.config.num_clients

        self.logger.info(f"Partitioning data for {num_clients} clients using '{self.partition_strategy}' strategy")

        partitions = partition_data_stratified(
                df,
                num_clients,
                self.constants.TARGET_COLUMN,
                self.config.seed,
                self.logger
            )

        # Log partition statistics
        for i, partition in enumerate(partitions):
            class_dist = partition[self.constants.TARGET_COLUMN].value_counts().to_dict()
            self.logger.info(f"Client {i}: {len(partition)} samples, class distribution: {class_dist}")

        return partitions

    def _run_federated_simulation(
        self,
        client_partitions: list,
        image_dir: str,
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        Run Flower federated learning simulation using SimulationRunner.

        Args:
            client_partitions: List of client data partitions
            image_dir: Directory containing images
            experiment_name: Name of experiment

        Returns:
            Dictionary with simulation results
        """
        self.logger.info("Starting Flower federated learning simulation...")

        try:
            # Run simulation using SimulationRunner
            results = self.simulation_runner.run_simulation(
                client_partitions=client_partitions,
                image_dir=image_dir,
                experiment_name=experiment_name
            )

            # Add additional metadata
            results['partition_strategy'] = self.partition_strategy
            results['checkpoint_dir'] = self.checkpoint_dir
            results['logs_dir'] = self.logs_dir

            return results

        except Exception as e:
            self.logger.error(f"Federated simulation failed: {e}")
            raise

    def train_from_zip(
        self,
        zip_path: str,
        experiment_name: str = "federated_pneumonia_detection",
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
                result = self.handler.validate_contents(source_path)
            elif os.path.isdir(source_path):
                result = self.handler.validate_contents(source_path)
            else:
                return {'valid': False, 'error': 'Path is neither a file nor a directory'}

            # Normalize result to dictionary format
            if isinstance(result, dict):
                return result
            elif isinstance(result, str):
                return {'valid': bool(result.strip()), 'error': result if not result.strip() else None}
            elif hasattr(result, '__bool__'):
                return {'valid': bool(result), 'error': None}
            else:
                return {'valid': True, 'error': None}

        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and configuration."""
        return {
            'checkpoint_dir': self.checkpoint_dir,
            'logs_dir': self.logs_dir,
            'partition_strategy': self.partition_strategy,
            'config': {
                'num_rounds': self.config.num_rounds,
                'num_clients': self.config.num_clients,
                'clients_per_round': self.config.clients_per_round,
                'local_epochs': self.config.local_epochs,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size
            },
            'temp_dir_active': self.handler.temp_extract_dir is not None
        }

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
