"""
Federated learning orchestrator for pneumonia detection.

This module coordinates the complete federated learning workflow from
data preparation through model training to final checkpoint saving.

Dependencies:
- .client: FlowerClient implementation
- data utilities: For data loading and partitioning
- Flower framework: For simulation

Role in System:
- Orchestrates end-to-end federated learning workflow
- Delegates specific tasks to specialized components
- Provides consistent API for training experiments
"""

import os
import logging
import torch
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from flwr.common import Context
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
from federated_pneumonia_detection.src.control.dl_model.utils.data import (
    DataSourceExtractor, DatasetPreparer
)
from federated_pneumonia_detection.src.control.federated_learning.client import FlowerClient
from federated_pneumonia_detection.src.control.federated_learning.server import create_federated_strategy
from federated_pneumonia_detection.src.control.federated_learning.partitioner import partition_data_stratified
from federated_pneumonia_detection.src.control.federated_learning.data_manager import ClientDataManager
from federated_pneumonia_detection.src.entities.resnet_with_custom_head import ResNetWithCustomHead


class FederatedTrainer:
    """Orchestrates federated learning workflow end-to-end."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_dir: str = "fed_results/checkpoints",
        logs_dir: str = "fed_results/logs",
    ):
        """
        Initialize federated trainer with configuration and directories.

        Args:
            config_path: Optional path to configuration file
            checkpoint_dir: Directory to save model checkpoints
            logs_dir: Directory to save training logs

        Raises:
            Exception: If configuration loading fails
        """
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.logger = self._setup_logging()

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

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
            self.logger.warning(f"Config loading failed: {e}. Using defaults.")
            self.constants = config_loader.create_system_constants()
            self.config = config_loader.create_experiment_config()

        # Initialize data utilities
        try:
            self.data_extractor = DataSourceExtractor(self.logger)
            self.dataset_preparer = DatasetPreparer(self.constants, self.config)
        except Exception as e:
            self.logger.error(f"Failed to initialize utilities: {e}")
            raise

        self.logger.info("FederatedTrainer initialized")

    def train(
        self,
        source_path: str,
        experiment_name: str = "federated_pneumonia_detection",
        csv_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute complete federated training workflow.

        Args:
            source_path: Path to zip file or directory containing dataset
            experiment_name: Name for this training experiment
            csv_filename: Optional specific CSV filename to look for

        Returns:
            Dictionary with training results and paths

        Raises:
            ValueError: If data validation fails
            Exception: If training fails
        """
        self.logger.info(f"Starting federated training from: {source_path}")

        try:
            # Extract and validate data
            image_dir, csv_path = self.data_extractor.extract_and_validate(
                source_path, csv_filename
            )

            # Load and prepare dataset
            train_df, val_df = self.dataset_preparer.prepare_dataset(csv_path, image_dir)

            # Combine for federated partitioning
            full_df = pd.concat([train_df, val_df], ignore_index=True)

            # Partition data across clients
            client_partitions = partition_data_stratified(
                full_df,
                self.config.num_clients,
                self.constants.TARGET_COLUMN,
                self.config.seed,
                self.logger
            )

            # Run federated simulation
            results = self._run_simulation(
                client_partitions,
                image_dir,
                experiment_name
            )

            return results

        except Exception as e:
            self.logger.error(f"Federated training failed: {e}")
            raise
        finally:
            self.data_extractor.cleanup()

    def _run_simulation(
        self,
        client_partitions: List[pd.DataFrame],
        image_dir: str,
        experiment_name: str,
    ) -> Dict[str, Any]:
        """
        Run Flower federated learning simulation.

        Args:
            client_partitions: List of client data partitions
            image_dir: Directory containing images
            experiment_name: Name of experiment

        Returns:
            Dictionary with simulation results

        Raises:
            ValueError: If partitions are empty
        """
        if not client_partitions or len(client_partitions) == 0:
            raise ValueError("client_partitions cannot be empty")

        num_clients = len(client_partitions)
        self.logger.info(
            f"Starting simulation: {num_clients} clients, {self.config.num_rounds} rounds"
        )

        # Create data manager
        data_manager = ClientDataManager(
            image_dir=image_dir,
            constants=self.constants,
            config=self.config,
            logger=self.logger,
        )

        # Prepare client dataloaders
        client_dataloaders = self._prepare_client_dataloaders(
            client_partitions, data_manager
        )

        # Create initial global model
        global_model = self._create_model()
        initial_parameters = [val.cpu().numpy() for val in global_model.state_dict().values()]

        # Create federated strategy
        strategy = self._create_strategy(initial_parameters, num_clients)

        # Configure server
        server_config = ServerConfig(num_rounds=self.config.num_rounds)

        # Run simulation
        self.logger.info("=" * 80)
        self.logger.info(f"Starting Flower simulation")
        self.logger.info(f"Rounds: {self.config.num_rounds}, Clients: {num_clients}")
        self.logger.info("=" * 80)

        try:
            client_fn = self._create_client_fn(client_dataloaders)

            history = start_simulation(
                client_fn=client_fn,
                num_clients=num_clients,
                client_resources={"num_cpus": 1, "num_gpus": 0.0},
                config=server_config,
                strategy=strategy,
            )

            self.logger.info("=" * 80)
            self.logger.info("Simulation completed successfully!")
            self.logger.info("=" * 80)

            # Process and save results
            results = self._process_results(history, experiment_name, num_clients)
            self._save_final_model(history, experiment_name)

            return results

        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error(f"Simulation failed: {type(e).__name__}: {str(e)}")
            self.logger.error("=" * 80)
            raise

    def _prepare_client_dataloaders(
        self,
        client_partitions: List[pd.DataFrame],
        data_manager: ClientDataManager,
    ) -> List[Tuple]:
        """
        Create DataLoaders for each client partition.

        Args:
            client_partitions: List of client data partitions
            data_manager: ClientDataManager instance

        Returns:
            List of (train_loader, val_loader) tuples

        Raises:
            Exception: If DataLoader creation fails
        """
        client_dataloaders = []

        for i, partition in enumerate(client_partitions):
            if len(partition) == 0:
                self.logger.warning(f"Client {i} has empty partition, skipping")
                continue

            try:
                train_loader, val_loader = data_manager.create_dataloaders_for_partition(
                    partition, self.config.validation_split
                )
                client_dataloaders.append((train_loader, val_loader))
            except Exception as e:
                self.logger.error(f"Failed to create DataLoaders for client {i}: {e}")
                raise

        if len(client_dataloaders) == 0:
            raise ValueError("No valid client dataloaders created")

        return client_dataloaders

    def _create_client_fn(
        self,
        client_dataloaders: List[Tuple],
    ):
        """
        Create client factory function with closure over dataloaders.

        Args:
            client_dataloaders: List of (train_loader, val_loader) tuples

        Returns:
            Callable that creates FlowerClient instances
        """
        def client_fn(context: Context):
            """Factory function for creating clients in Flower simulation."""
            client_id = int(context.node_id)

            if client_id >= len(client_dataloaders):
                raise ValueError(f"Client {client_id} out of range")

            train_loader, val_loader = client_dataloaders[client_id]

            # Create model for this client
            model = self._create_model()

            # Create device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Create Flower client
            flower_client = FlowerClient(
                client_id=client_id,
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                device=device,
                config=self.config,
                logger=self.logger,
            )

            # Convert to Flower Client
            return flower_client.to_client()

        return client_fn

    def _create_model(self) -> ResNetWithCustomHead:
        """
        Create ResNetWithCustomHead model instance.

        Returns:
            Initialized model
        """
        model = ResNetWithCustomHead(
            constants=self.constants,
            config=self.config,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            fine_tune_layers_count=self.config.fine_tune_layers_count,
        )
        return model

    def _create_strategy(
        self,
        initial_parameters: List,
        num_clients: int,
    ) -> FedAvg:
        """Create federated averaging strategy."""
        clients_per_round = min(self.config.clients_per_round, num_clients)
        return create_federated_strategy(num_clients, clients_per_round, initial_parameters)

    def _process_results(
        self,
        history: Any,
        experiment_name: str,
        num_clients: int,
    ) -> Dict[str, Any]:
        """
        Process simulation history into results dictionary.

        Args:
            history: Flower simulation history
            experiment_name: Name of experiment
            num_clients: Number of clients

        Returns:
            Formatted results dictionary
        """
        results = {
            "experiment_name": experiment_name,
            "num_clients": num_clients,
            "num_rounds": self.config.num_rounds,
            "status": "completed",
            "checkpoint_dir": self.checkpoint_dir,
            "logs_dir": self.logs_dir,
            "metrics": {
                "losses_distributed": getattr(history, "losses_distributed", []),
                "metrics_distributed": getattr(history, "metrics_distributed", []),
                "losses_centralized": getattr(history, "losses_centralized", []),
                "metrics_centralized": getattr(history, "metrics_centralized", []),
            },
        }

        # Log final loss if available
        if hasattr(history, "losses_distributed") and history.losses_distributed:
            final_loss = history.losses_distributed[-1][1]
            self.logger.info(f"Final distributed loss: {final_loss:.4f}")

        return results

    def _save_final_model(self, history: Any, experiment_name: str) -> None:
        """
        Save final model parameters to checkpoint.

        Args:
            history: Flower simulation history
            experiment_name: Name of experiment
        """
        try:
            final_parameters = None

            if hasattr(history, "parameters_distributed") and history.parameters_distributed:
                final_parameters = history.parameters_distributed[-1][0]
            elif hasattr(history, "parameters") and history.parameters:
                final_parameters = history.parameters[-1]

            if final_parameters is None:
                self.logger.warning("No parameters found in history - skipping model save")
                return

            # Create model and load parameters
            model = self._create_model()
            params_dict = zip(model.state_dict().keys(), final_parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)

            # Save checkpoint
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"{experiment_name}_final_model.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)

            self.logger.info(f"Final model saved to: {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Failed to save final model: {e}")

    def _setup_logging(self) -> logging.Logger:
        """
        Configure logger for training.

        Returns:
            Configured logger instance
        """
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
