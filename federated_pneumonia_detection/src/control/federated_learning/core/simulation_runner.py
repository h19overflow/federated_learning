"""
Federated learning simulation runner.
Orchestrates Flower simulation with client functions and data partitions.
This is the bridge between data partitions and Flower's federated training.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import torch
from collections import OrderedDict

from flwr.client import ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.entities.resnet_with_custom_head import ResNetWithCustomHead
from federated_pneumonia_detection.src.control.federated_learning.data.client_data import ClientDataManager
from federated_pneumonia_detection.src.control.federated_learning.training.functions import (
    get_model_parameters,
    set_model_parameters,
)
from federated_pneumonia_detection.src.control.federated_learning.core.fed_client import FlowerClient



class SimulationRunner:
    """
    Orchestrates Flower federated learning simulation.
    Connects data partitions with Flower clients and runs training.
    """

    def __init__(
        self,
        constants: SystemConstants,
        config: ExperimentConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize simulation runner.

        Args:
            constants: SystemConstants for configuration
            config: ExperimentConfig for training parameters
            logger: Optional logger instance
        """
        self.constants = constants
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info("SimulationRunner initialized")

    def run_simulation(
        self,
        client_partitions: List[pd.DataFrame],
        image_dir: str,
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        Run Flower federated learning simulation.

        Args:
            client_partitions: List of DataFrames, one per client
            image_dir: Directory containing images
            experiment_name: Name of experiment

        Returns:
            Dictionary with simulation results

        Raises:
            ValueError: If partitions are invalid
        """
        if not client_partitions or len(client_partitions) == 0:
            raise ValueError("client_partitions cannot be empty")

        num_clients = len(client_partitions)
        self.logger.info(f"Starting FL simulation: {num_clients} clients, {self.config.num_rounds} rounds")

        # Create data manager
        data_manager = ClientDataManager(
            image_dir=image_dir,
            constants=self.constants,
            config=self.config,
            logger=self.logger
        )

        # Prepare client data loaders
        client_dataloaders = []
        for i, partition in enumerate(client_partitions):
            if len(partition) == 0:
                self.logger.warning(f"Client {i} has empty partition, skipping")
                continue

            try:
                train_loader, val_loader = data_manager.create_dataloaders_for_partition(partition)
                client_dataloaders.append((train_loader, val_loader))
                self.logger.info(
                    f"Client {i}: {len(train_loader.dataset)} train, "
                    f"{len(val_loader.dataset)} val samples"
                )
            except Exception as e:
                self.logger.error(f"Failed to create DataLoaders for client {i}: {e}")
                raise

        if len(client_dataloaders) == 0:
            raise ValueError("No valid client dataloaders created")

        # Create client function factory
        def client_fn(cid: str) -> FlowerClient:
            """Create a Flower client for the given client ID."""
            client_id = int(cid)
            train_loader, val_loader = client_dataloaders[client_id]

            return FlowerClient(
                client_id=client_id,
                train_loader=train_loader,
                val_loader=val_loader,
                constants=self.constants,
                config=self.config,
                logger=self.logger
            )

        # Create initial global model
        self.logger.info("Initializing global model...")
        global_model = ResNetWithCustomHead(
            constants=self.constants,
            config=self.config,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            fine_tune_layers_count=self.config.fine_tune_layers_count
        )

        initial_parameters = get_model_parameters(global_model)

        # Create FedAvg strategy
        fraction_fit = float(self.config.clients_per_round) / num_clients
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,  # Evaluate on all clients
            min_fit_clients=min(self.config.clients_per_round, num_clients),
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters
        )

        # Configure simulation
        client_resources = {
            "num_cpus": 1,
            "num_gpus": 0.0  # Adjust based on available GPUs
        }

        # Run simulation
        self.logger.info(f"Starting simulation: {self.config.num_rounds} rounds")

        try:
            history = run_simulation(
                client_fn=client_fn,
                num_clients=len(client_dataloaders),
                config=ServerConfig(num_rounds=self.config.num_rounds),
                strategy=strategy,
                client_resources=client_resources
            )

            self.logger.info("Simulation completed successfully!")

            # Extract results
            results = self._process_simulation_results(history, experiment_name, num_clients)

            # Save final model
            if history.parameters_distributed:
                final_parameters = history.parameters_distributed[-1][0]
                self._save_final_model(final_parameters, experiment_name)

            return results

        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise

    def _process_simulation_results(
        self,
        history,
        experiment_name: str,
        num_clients: int
    ) -> Dict[str, Any]:
        """
        Process simulation history into results dictionary.

        Args:
            history: Flower simulation history
            experiment_name: Name of experiment
            num_clients: Number of clients

        Returns:
            Dictionary with formatted results
        """
        results = {
            'experiment_name': experiment_name,
            'num_clients': num_clients,
            'num_rounds': self.config.num_rounds,
            'status': 'completed',
            'metrics': {
                'losses_distributed': history.losses_distributed,
                'metrics_distributed': history.metrics_distributed,
                'losses_centralized': history.losses_centralized,
                'metrics_centralized': history.metrics_centralized
            }
        }

        # Log summary
        if history.losses_distributed:
            final_loss = history.losses_distributed[-1][1]
            self.logger.info(f"Final distributed loss: {final_loss:.4f}")

        return results

    def _save_final_model(self, parameters: List, experiment_name: str) -> None:
        """
        Save final model parameters to checkpoint.

        Args:
            parameters: Final model parameters
            experiment_name: Name of experiment
        """
        import os

        # Create model and load parameters
        model = ResNetWithCustomHead(
            constants=self.constants,
            config=self.config,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            fine_tune_layers_count=self.config.fine_tune_layers_count
        )

        set_model_parameters(model, parameters)

        # Save checkpoint
        checkpoint_dir = getattr(self.config, 'checkpoint_dir', 'federated_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_final_model.pt")
        torch.save(model.state_dict(), checkpoint_path)

        self.logger.info(f"Final model saved to: {checkpoint_path}")
