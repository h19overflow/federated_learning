"""
Federated learning simulation runner.
Orchestrates Flower simulation with client functions and data partitions.
This is the bridge between data partitions and Flower's federated training.

Note: This implementation uses start_simulation() with a client_fn factory.
The client_fn creates FlowerClient (NumPyClient) instances and converts them
to Client objects using .to_client(), which is compatible with the modern
Flower API (1.0+).

For deployment scenarios (non-simulation), consider using the ClientApp pattern
defined in client_app.py and server_app.py instead.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import torch

from flwr.client import NumPyClient, Client
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from flwr.common import Context

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
    
    This class uses Flower's start_simulation() API, which is designed for
    simulating federated learning locally. It creates a client_fn factory that:
    
    1. Takes a Context object with node_id
    2. Creates a FlowerClient (NumPyClient subclass) instance
    3. Converts it to a Client using .to_client()
    4. Returns the Client for Flower's simulation engine
    
    The FlowerClient class (in fed_client.py) is a NumPyClient that implements:
    - get_parameters(): Returns model parameters as numpy arrays
    - set_parameters(): Sets model parameters from numpy arrays  
    - fit(): Trains model locally and returns updated parameters
    - evaluate(): Evaluates model and returns metrics
    
    This approach is compatible with both Flower's simulation API and the
    modern ClientApp deployment pattern. The same FlowerClient can be used in:
    - Simulation: via start_simulation() (this file)
    - Deployment: via ClientApp with client_fn (client_app.py)
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
        self.logger = logger or logging.getLogger(__name__, level=logging.warning)

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
        def client_fn(context: Context) -> Client:
            """Create a Flower client for the given context."""
            try:
                # Get client ID from context (for start_simulation, use node_id or cid)
                # Context.node_id contains the client ID as string
                client_id = int(context.node_id)
                self.logger.debug(f"Creating client for node_id={context.node_id} (client_id={client_id})")
                
                if client_id >= len(client_dataloaders):
                    self.logger.error(f"Client ID {client_id} out of range (max: {len(client_dataloaders)-1})")
                    raise ValueError(f"Client ID {client_id} exceeds available dataloaders")
                
                train_loader, val_loader = client_dataloaders[client_id]
                self.logger.debug(
                    f"Client {client_id} dataloaders: "
                    f"train={len(train_loader.dataset)} samples, "
                    f"val={len(val_loader.dataset)} samples"
                )

                flower_client = FlowerClient(
                    client_id=client_id,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    constants=self.constants,
                    config=self.config,
                    logger=self.logger
                )
                
                self.logger.debug(f"FlowerClient {client_id} created successfully")
                return flower_client.to_client()
                
            except Exception as e:
                self.logger.error(f"CRITICAL ERROR in client_fn for context {context.node_id}: {type(e).__name__}: {str(e)}")
                import traceback
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise

        # Create initial global model
        self.logger.info("Initializing global model...")
        try:
            global_model = ResNetWithCustomHead(
                constants=self.constants,
                config=self.config,
                num_classes=self.config.num_classes,
                dropout_rate=self.config.dropout_rate,
                fine_tune_layers_count=self.config.fine_tune_layers_count
            )
            self.logger.info("Global model created successfully")
            
            self.logger.debug("Extracting initial model parameters...")
            initial_parameters = get_model_parameters(global_model)
            self.logger.info(f"Initial parameters extracted: {len(initial_parameters)} arrays")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize global model: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

        # Create FedAvg strategy
        fraction_fit = float(self.config.clients_per_round) / num_clients
        self.logger.info(f"Creating FedAvg strategy with fraction_fit={fraction_fit:.2f}")
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,  # Evaluate on all clients
            min_fit_clients=min(self.config.clients_per_round, num_clients),
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters
        )
        self.logger.info("FedAvg strategy created successfully")

        # Configure client resources
        client_resources = {
            "num_cpus": 1,
            "num_gpus": 0.0  # Adjust based on available GPUs
        }

        # Run simulation
        self.logger.info("="*80)
        self.logger.info(f"Starting Flower simulation: {self.config.num_rounds} rounds with {len(client_dataloaders)} clients")
        self.logger.info(f"Client resources: {client_resources}")
        self.logger.info("="*80)

        try:
            history = start_simulation(
                client_fn=client_fn,
                num_clients=len(client_dataloaders),
                client_resources=client_resources,
                config=ServerConfig(num_rounds=self.config.num_rounds),
                strategy=strategy
            )

            self.logger.info("="*80)
            self.logger.info("Simulation completed successfully!")
            self.logger.info("="*80)
            
            # Log history attributes for debugging
            self.logger.debug(f"History object type: {type(history)}")
            self.logger.debug(f"History attributes: {dir(history)}")

            # Extract results
            results = self._process_simulation_results(history, experiment_name, num_clients)

            # Save final model - handle different Flower versions
            # Check if history has parameters_distributed (older Flower) or parameters (newer Flower)
            if hasattr(history, 'parameters_distributed') and history.parameters_distributed:
                self.logger.info("Saving final model using parameters_distributed...")
                final_parameters = history.parameters_distributed[-1][0]
                self._save_final_model(final_parameters, experiment_name)
            elif hasattr(history, 'parameters') and history.parameters:
                self.logger.info("Saving final model using parameters...")
                final_parameters = history.parameters[-1]
                self._save_final_model(final_parameters, experiment_name)
            else:
                self.logger.warning("No parameters found in history - skipping model save")
                self.logger.debug(f"History has: {[attr for attr in dir(history) if not attr.startswith('_')]}")

            return results

        except Exception as e:
            self.logger.error("="*80)
            self.logger.error(f"Simulation failed: {type(e).__name__}: {str(e)}")
            self.logger.error("="*80)
            import traceback
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
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
        if hasattr(history, 'losses_distributed') and history.losses_distributed:
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
