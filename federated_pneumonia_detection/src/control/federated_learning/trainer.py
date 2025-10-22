"""
Federated Learning Trainer orchestrating the FL simulation.

Manages the entire federated learning pipeline: data loading, partitioning,
client creation, and Flower simulation execution.

Dependencies:
- Flower: Federated learning framework
- PyTorch: Deep learning
- pandas: Data manipulation
- torch: Model training

Role in System:
- Loads and partitions training data
- Creates FlowerClient instances for each client
- Initializes Flower server strategy (FedAvg)
- Executes federated learning simulation
- Returns training history and metrics
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable

import torch
import pandas as pd
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, Parameters

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.entities.resnet_with_custom_head import (
    ResNetWithCustomHead,
)
from federated_pneumonia_detection.src.control.federated_learning.client import (
    FlowerClient,
)
from federated_pneumonia_detection.src.control.federated_learning.data_manager import (
    load_data,
)
from federated_pneumonia_detection.src.control.federated_learning.partitioner import (
    partition_data_stratified,
)


logger = logging.getLogger(__name__)


class FederatedTrainer:
    """Orchestrate federated learning training pipeline."""

    def __init__(
        self,
        config: ExperimentConfig,
        constants: SystemConstants,
        device: torch.device,
        websocket_manager: Any = None,
    ) -> None:
        """
        Initialize FederatedTrainer.

        Args:
            config: ExperimentConfig with FL hyperparameters
            constants: SystemConstants for data and model paths
            device: torch.device for GPU/CPU selection
            websocket_manager: Optional WebSocket manager for real-time progress updates
        """
        if config is None:
            raise ValueError("config cannot be None")
        if constants is None:
            raise ValueError("constants cannot be None")
        if device is None:
            raise ValueError("device cannot be None")

        self.config = config
        self.constants = constants
        self.device = device
        self.websocket_manager = websocket_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client_instances = {}
        self.metrics_dir = None
        self.experiment_name = None

    def _create_model(self) -> ResNetWithCustomHead:
        """Create a ResNetWithCustomHead model instance."""
        model = ResNetWithCustomHead(
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            fine_tune_layers_count=self.config.fine_tune_layers_count,
            constants=self.constants,
            config=self.config,
        )
        return model.to(self.device)

    def _get_initial_parameters(self) -> Parameters:
        """
        Create initial model and extract parameters for server.

        Returns:
            Flower Parameters object with initial model weights
        """
        model = self._create_model()
        weights = [val.cpu().numpy() for val in model.state_dict().values()]
        del model
        return ndarrays_to_parameters(weights)

    def _client_fn(
        self, cid: str
    ) -> FlowerClient:
        """
        Factory function to create a FlowerClient for a given client ID.

        Args:
            cid: Client ID

        Returns:
            FlowerClient instance

        Note:
            This is called by Flower's start_simulation for each client.
            The DataFrames and dataloaders should be pre-loaded and available.
        """
        if cid not in self._client_data_cache:
            raise ValueError(f"Client {cid} data not found in cache")

        train_loader, val_loader = self._client_data_cache[cid]

        model = self._create_model()

        client = FlowerClient(
            net=model,
            trainloader=train_loader,
            valloader=val_loader,
            config=self.config,
            device=self.device,
            client_id=cid,
            metrics_dir=self.metrics_dir,
            experiment_name=self.experiment_name,
            websocket_manager=self.websocket_manager,
        )
        
        # Store client reference for later finalization
        self._client_instances[cid] = client
        
        return client

    def _finalize_client_metrics(self) -> Dict[str, Any]:
        """
        Finalize all client metrics, save individual files, and aggregate into single file.

        Returns:
            Aggregated metrics dictionary with per-client data
        """
        aggregated_metrics = {
            "experiment_name": self.experiment_name,
            "num_clients": len(self._client_instances),
            "num_rounds": self.config.num_rounds,
            "clients": {}
        }

        # Finalize each client and collect their metrics
        for client_id, client in self._client_instances.items():
            try:
                client.finalize()
                
                # Collect client metrics if available
                if hasattr(client, 'metrics_collector') and client.metrics_collector:
                    aggregated_metrics["clients"][client_id] = {
                        "metadata": client.metrics_collector.get_metadata(),
                        "round_metrics": client.metrics_collector.get_round_metrics(),
                        "local_epoch_metrics": client.metrics_collector.get_local_epoch_metrics(),
                    }
            except Exception as e:
                self.logger.error(f"Error finalizing client {client_id}: {e}")

        return aggregated_metrics

    def _save_aggregated_client_metrics(self, aggregated_metrics: Dict[str, Any], filename: str = "fl_clients_history.json"):
        """
        Save aggregated client metrics to JSON file.

        Args:
            aggregated_metrics: Aggregated metrics from all clients
            filename: Output filename
        """
        output_path = Path(self.metrics_dir) / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(aggregated_metrics, f, indent=2)
            self.logger.info(f"Saved aggregated client metrics to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save aggregated client metrics: {e}")

    def _create_evaluate_fn(
        self, val_loader
    ) -> Callable[[int, List, Dict[str, Any]], Tuple[float, Dict[str, Any]]]:
        """
        Create server-side evaluation function for global model validation.

        Args:
            val_loader: Validation DataLoader for global evaluation

        Returns:
            Evaluation function for Flower strategy
        """

        def evaluate_fn(
            server_round: int, parameters: List, config: Dict[str, Any]
        ) -> Tuple[float, Dict[str, Any]]:
            """
            Evaluate global model on validation set.

            Args:
                server_round: Current round number
                parameters: Global model parameters
                config: Config dict from server

            Returns:
                Tuple of (loss, metrics_dict)
            """
            model = self._create_model()

            # Load global parameters into model
            params_dict = zip(model.state_dict().keys(), parameters)
            from collections import OrderedDict
            state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in params_dict}
            )
            model.load_state_dict(state_dict, strict=True)

            # Evaluate on validation set
            import torch.nn as nn

            criterion = (
                nn.BCEWithLogitsLoss()
                if self.config.num_classes == 1
                else nn.CrossEntropyLoss()
            )
            model.eval()
            correct, total_loss = 0, 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(
                        self.device
                    )
                    outputs = model(images)

                    if self.config.num_classes == 1:
                        labels_for_loss = labels.float().unsqueeze(1)
                        total_loss += criterion(outputs, labels_for_loss).item()
                        predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                        correct += (predictions == labels).sum().item()
                    else:
                        total_loss += criterion(outputs, labels.long()).item()
                        _, predictions = torch.max(outputs, 1)
                        correct += (predictions == labels).sum().item()

            accuracy = correct / len(val_loader.dataset)
            avg_loss = total_loss / len(val_loader)

            self.logger.info(
                f"Round {server_round}: Global Val Loss={avg_loss:.4f}, "
                f"Global Val Accuracy={accuracy:.4f}"
            )

            return avg_loss, {"accuracy": accuracy}

        return evaluate_fn

    def _save_history(
        self, history: Any, filename: str = "fl_history.json"
    ) -> None:
        """
        Save the Flower History object to a JSON file.

        Args:
            history: Flower History object containing training metrics
            filename: Output filename for the history JSON file

        Raises:
            Exception: If saving fails
        """
        try:
            history_data = {
                "losses_distributed": history.losses_distributed,
                "losses_centralized": history.losses_centralized,
                "metrics_distributed": history.metrics_distributed,
                "metrics_centralized": history.metrics_centralized,
                "metrics_distributed_fit": history.metrics_distributed_fit,
            }

            with open(filename, "w") as f:
                json.dump(history_data, f, indent=2)

            self.logger.info(f"History saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save history to {filename}: {e}")
            raise

    def train(
        self,
        data_df: pd.DataFrame,
        image_dir: Path,
        experiment_name: str = "federated_learning",
    ) -> Dict[str, Any]:
        """
        Execute federated learning training.

        Steps:
        1. Partition data across clients
        2. Create client dataloaders
        3. Get initial parameters
        4. Configure FedAvg strategy with evaluation
        5. Run Flower simulation
        6. Return history

        Args:
            data_df: DataFrame with filename and target columns
            image_dir: Directory containing training images
            experiment_name: Name for the experiment

        Returns:
            Dictionary with training results and metrics

        Raises:
            ValueError: If data_df is empty or invalid
            RuntimeError: If training fails
        """
        self.logger.info("="*80)
        self.logger.info(f"Starting Federated Learning: {experiment_name}")
        self.logger.info("="*80)

        try:
            # 1. Partition data
            self.logger.info(
                f"\nPartitioning data into {self.config.num_clients} clients..."
            )
            client_partitions = partition_data_stratified(
                df=data_df,
                num_clients=self.config.num_clients,
                target_column=self.constants.TARGET_COLUMN,
                seed=self.config.seed,
            )

            # Log partition statistics
            for idx, partition in enumerate(client_partitions):
                class_dist = partition[
                    self.constants.TARGET_COLUMN
                ].value_counts().to_dict()
                self.logger.info(
                    f"  Client {idx}: {len(partition)} samples, "
                    f"class distribution: {class_dist}"
                )

            # 2. Create dataloaders and cache them for client_fn
            self.logger.info("\nCreating client dataloaders...")
            self._client_data_cache: Dict[str, Tuple] = {}

            for client_id, partition_df in enumerate(client_partitions):
                train_loader, val_loader = load_data(
                    partition_df=partition_df,
                    image_dir=image_dir,
                    constants=self.constants,
                    config=self.config,
                )
                self._client_data_cache[str(client_id)] = (
                    train_loader,
                    val_loader,
                )
                self.logger.info(
                    f"  Client {client_id}: "
                    f"{len(train_loader.dataset)} train, "
                    f"{len(val_loader.dataset)} val"
                )

            # 3. Get global validation data (reserved data not used by any client)
            # For simplicity, use a portion of the last client's validation set
            _, global_val_loader = load_data(
                partition_df=client_partitions[0],
                image_dir=image_dir,
                constants=self.constants,
                config=self.config,
            )

            # 4. Get initial parameters
            self.logger.info("\nInitializing global model parameters...")
            initial_parameters = self._get_initial_parameters()

            # 5. Create FedAvg strategy with server-side evaluation
            self.logger.info("\nConfiguring FedAvg strategy...")
            strategy = FedAvg(
                fraction_fit=1.0,
                fraction_evaluate=0.0,
                min_fit_clients=self.config.num_clients,
                min_evaluate_clients=0,
                min_available_clients=self.config.num_clients,
                evaluate_fn=self._create_evaluate_fn(global_val_loader),
                initial_parameters=initial_parameters,
                fit_metrics_aggregation_fn=weighted_average,
            )

            # 6. Configure client resources and metrics collection
            client_resources = {
                "num_gpus": 1.0 if self.device.type == "cuda" else 0.0,
                "num_cpus": 1.0,
            }
            
            # Set metrics directory and experiment name for clients
            self.experiment_name = experiment_name
            self.metrics_dir = Path("results/federated_learning/metrics")
            self.metrics_dir.mkdir(parents=True, exist_ok=True)

            # 7. Run simulation
            self.logger.info("\nStarting Flower simulation...")
            self.logger.info(
                f"  Num rounds: {self.config.num_rounds}"
            )
            self.logger.info(f"  Num clients: {self.config.num_clients}")
            self.logger.info(
                f"  Local epochs per round: {self.config.local_epochs}"
            )
            self.logger.info("-"*80)

            history = fl.simulation.start_simulation(
                client_fn=self._client_fn,
                num_clients=self.config.num_clients,
                config=fl.server.ServerConfig(num_rounds=self.config.num_rounds),
                strategy=strategy,
                client_resources=client_resources,
            )

            # 8. Extract results and finalize client metrics
            self.logger.info("\n" + "-"*80)
            self.logger.info("Federated Learning Completed!")
            self.logger.info("Finalizing client metrics...")
            
            aggregated_client_metrics = self._finalize_client_metrics()
            self._save_aggregated_client_metrics(aggregated_client_metrics)

            results = {
                "experiment_name": experiment_name,
                "status": "completed",
                "num_clients": self.config.num_clients,
                "num_rounds": self.config.num_rounds,
                "local_epochs": self.config.local_epochs,
                "history": history,
                "metrics": {
                    "losses_distributed": history.losses_distributed,
                    "metrics_distributed": history.metrics_distributed,
                    "metrics_centerlized": history.metrics_centralized,
                    "losses_centralized": history.losses_centralized,
                },
            }
            
            # Save history to JSON
            self._save_history(history, "fl_history.json")
            return results

        except Exception as e:
            self.logger.error(f"Federated training failed: {e}", exc_info=True)
            raise RuntimeError(f"Training failed: {e}") from e

# Helper function to aggregate metrics
def weighted_average(metrics):
    # metrics: List of tuples (num_examples, metrics_dict)
    total_examples = sum([num_examples for num_examples, _ in metrics])
    result = {}
    for num_examples, m in metrics:
        for k, v in m.items():
            if k not in result:
                result[k] = 0.0
            result[k] += v * num_examples / total_examples
    return result