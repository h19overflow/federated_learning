"""
Federated training orchestrator for pneumonia detection system.
Mirrors CentralizedTrainer API while orchestrating federated learning workflow.

This module uses Flower's ClientApp pattern:
1. FlowerClient (NumPyClient) - implements fit() and evaluate() for local training
2. client_fn() - factory that creates FlowerClient instances and converts to Client via .to_client()
3. ClientApp - wraps client_fn for Flower framework
4. start_simulation() - runs federated learning simulation locally
"""

import os
import logging
import torch
from typing import Optional, Dict, Any, List
import pandas as pd

from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from federated_pneumonia_detection.src.utils.config_loader import ConfigLoader
from federated_pneumonia_detection.src.control.dl_model.utils.data import (
    DataSourceExtractor, DatasetPreparer
)
from federated_pneumonia_detection.src.control.federated_learning.data.partitioner import (
    partition_data_stratified
)
from federated_pneumonia_detection.src.control.federated_learning.data.client_data import ClientDataManager
from federated_pneumonia_detection.src.control.federated_learning.core.fed_client import FlowerClient
from federated_pneumonia_detection.src.entities.resnet_with_custom_head import ResNetWithCustomHead
from federated_pneumonia_detection.src.control.federated_learning.training.functions import (
    get_model_parameters,
    set_model_parameters,
)


class FederatedTrainer:
    """
    Federated training orchestrator that handles complete FL workflow.
    Mirrors CentralizedTrainer API for consistent interface.
    
    This class orchestrates federated learning using Flower's modern ClientApp pattern:
    
    Architecture:
    1. FlowerClient (in fed_client.py) - extends NumPyClient with:
       - fit(): performs local training, returns updated parameters
       - evaluate(): evaluates model, returns metrics
       - Inherits .to_client() from NumPyClient to convert to Client
    
    2. client_fn() - factory function that:
       - Takes Context with node_id
       - Creates FlowerClient with appropriate data loaders
       - Returns flower_client.to_client() (Client instance)
    
    3. ClientApp - wraps client_fn for Flower framework
    
    4. start_simulation() - runs all clients locally for testing
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

        self.logger.info("FederatedTrainer initialized")
        
        # These will be set during training for client_fn access
        self.client_dataloaders = None
        self.data_manager = None

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

        try:
            # Detect source type and extract/find data
            image_dir, csv_path = self.handler.extract_and_validate(source_path,
                                                                    csv_filename)

            # Load and process data
            train_df, val_df = self.dataset_preparer.prepare_dataset(csv_path,
                                                                     image_dir)

            # Combine train and val for federated partitioning
            full_df = pd.concat([train_df, val_df],
                                ignore_index=True)

            # # Partition data across clients
            # client_partitions = self._partition_data_for_clients(full_df)
            client_partitions=partition_data_stratified(full_df, self.config.num_clients,
                                                        self.constants.TARGET_COLUMN,
                                                        self.config.seed,
                                                        self.logger)
            # Run federated learning simulation
            results = self._run_federated_simulation(
                client_partitions,
                image_dir,
                experiment_name
            )

            return results

        except Exception as e:
            self.logger.error(f"Federated training failed: {e}")
            raise
        finally:
            self.handler.cleanup()


    def _run_federated_simulation(
        self,
        client_partitions: List[pd.DataFrame],
        image_dir: str,
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        Run Flower federated learning simulation.
        
        This method implements the complete simulation workflow:
        1. Create data loaders for each client partition
        2. Define client_fn factory that creates FlowerClient instances
        3. Create ClientApp wrapping the client_fn
        4. Initialize global model and strategy
        5. Run start_simulation() with ClientApp
        6. Process and return results

        Args:
            client_partitions: List of client data partitions (DataFrames)
            image_dir: Directory containing images
            experiment_name: Name of experiment

        Returns:
            Dictionary with simulation results
        """
        if not client_partitions or len(client_partitions) == 0:
            raise ValueError("client_partitions cannot be empty")

        num_clients = len(client_partitions)
        self.logger.info(f"Starting FL simulation: {num_clients} clients, {self.config.num_rounds} rounds")

        # Create data manager
        self.data_manager = ClientDataManager(
            image_dir=image_dir,
            constants=self.constants,
            config=self.config,
            logger=self.logger
        )

        # Prepare client data loaders
        self.client_dataloaders = []
        for i, partition in enumerate(client_partitions):
            if len(partition) == 0:
                self.logger.warning(f"Client {i} has empty partition, skipping")
                continue

            try:
                train_loader, val_loader = self.data_manager.create_dataloaders_for_partition(partition)
                self.client_dataloaders.append((train_loader, val_loader))
                self.logger.info(
                    f"Client {i}: {len(train_loader.dataset)} train, "
                    f"{len(val_loader.dataset)} val samples"
                )
            except Exception as e:
                self.logger.error(f"Failed to create DataLoaders for client {i}: {e}")
                raise

        if len(self.client_dataloaders) == 0:
            raise ValueError("No valid client dataloaders created")

        # Create client function for ClientApp
        def client_fn(context: Context):
            """
            Client factory function that creates FlowerClient instances.
            
            This is called by Flower for each client node. It:
            1. Extracts client_id from context.node_id
            2. Gets the appropriate data loaders for this client
            3. Creates a FlowerClient (NumPyClient subclass)
            4. Converts to Client using .to_client() (inherited from NumPyClient)
            5. Returns the Client instance to Flower
            
            Args:
                context: Flower Context containing node_id and config
                
            Returns:
                Client: Flower Client instance ready for federated learning
            """
            try:
                # Extract client ID from context
                client_id = int(context.node_id)
                self.logger.debug(f"Creating client for node_id={context.node_id} (client_id={client_id})")
                
                if client_id >= len(self.client_dataloaders):
                    self.logger.error(f"Client ID {client_id} out of range (max: {len(self.client_dataloaders)-1})")
                    raise ValueError(f"Invalid client ID: {client_id}")

                # Get train and val loaders for this client
                train_loader, val_loader = self.client_dataloaders[client_id]

                # Create FlowerClient instance (NumPyClient subclass)
                flower_client = FlowerClient(
                    client_id=client_id,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    constants=self.constants,
                    config=self.config,
                    logger=self.logger
                )

                # Convert NumPyClient to Client using inherited .to_client() method
                # This is required by Flower's simulation/deployment framework
                return flower_client.to_client()
                
            except Exception as e:
                self.logger.error(
                    f"CRITICAL ERROR creating client for node_id={context.node_id}: "
                    f"{type(e).__name__}: {str(e)}"
                )
                import traceback
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise

        # Create ClientApp (modern Flower pattern)
        client_app = ClientApp(client_fn=client_fn)

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
        self.logger.info(f"Global model created with {len(initial_parameters)} parameter arrays")

        # Create federated averaging strategy
        strategy = FedAvg(
            fraction_fit=self.config.clients_per_round / len(self.client_dataloaders),
            fraction_evaluate=self.config.clients_per_round / len(self.client_dataloaders),
            min_fit_clients=self.config.clients_per_round,
            min_evaluate_clients=self.config.clients_per_round,
            min_available_clients=len(self.client_dataloaders),
            initial_parameters=initial_parameters,
        )

        # Configure server
        server_config = ServerConfig(num_rounds=self.config.num_rounds)

        # Set client resources (adjust based on your hardware)
        client_resources = {
            "num_cpus": 1,
            "num_gpus": 0.0  # Set to fraction like 0.25 if you have GPUs
        }

        # Run simulation
        self.logger.info("="*80)
        self.logger.info(f"Starting Flower simulation with ClientApp pattern")
        self.logger.info(f"Rounds: {self.config.num_rounds}, Clients: {len(self.client_dataloaders)}")
        self.logger.info(f"Client resources: {client_resources}")
        self.logger.info("="*80)

        try:
            history = start_simulation(
                client_fn=client_fn,  # Can pass client_fn directly or via client_app
                num_clients=len(self.client_dataloaders),
                client_resources=client_resources,
                config=server_config,
                strategy=strategy
            )

            self.logger.info("="*80)
            self.logger.info("Simulation completed successfully!")
            self.logger.info("="*80)

            # Process results
            results = self._process_simulation_results(history, experiment_name, num_clients)

            # Save final model
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
            'partition_strategy': self.partition_strategy,
            'checkpoint_dir': self.checkpoint_dir,
            'logs_dir': self.logs_dir,
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
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{experiment_name}_final_model.pt")
        torch.save(model.state_dict(), checkpoint_path)

        self.logger.info(f"Final model saved to: {checkpoint_path}")


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
