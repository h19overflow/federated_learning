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
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional
from datetime import datetime

import torch
import pandas as pd
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, Parameters, FitRes, FitIns
from flwr.server.client_proxy import ClientProxy
from sqlalchemy.orm import Session

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
    split_partition,
)

from federated_pneumonia_detection.src.control.dl_model.utils.data.websocket_metrics_sender import (
    MetricsWebSocketSender,
)
from federated_pneumonia_detection.src.boundary.engine import get_session
from federated_pneumonia_detection.src.boundary.CRUD.run import run_crud
from federated_pneumonia_detection.src.boundary.CRUD.client import ClientCRUD

logger = logging.getLogger(__name__)


class RoundTrackingFedAvg(FedAvg):
    """Custom FedAvg strategy that tracks the current server round."""
    
    def __init__(self, trainer_ref, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer_ref = trainer_ref
    
    def configure_fit(
        self, server_round: int, parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training and update trainer's round."""
        self.trainer_ref.current_server_round = server_round
        return super().configure_fit(server_round, parameters, client_manager)


class FederatedTrainer:
    """Orchestrate federated learning training pipeline."""

    def __init__(
        self,
        config: ExperimentConfig,
        constants: SystemConstants,
        device: torch.device,
        websocket_uri: Optional[str] = "ws://localhost:8765",
    ) -> None:
        """
        Initialize FederatedTrainer.

        Args:
            config: ExperimentConfig with FL hyperparameters
            constants: SystemConstants for data and model paths
            device: torch.device for GPU/CPU selection
            websocket_uri: WebSocket URI for real-time metrics streaming
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
        self.websocket_uri = websocket_uri
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client_instances: Dict[str, FlowerClient] = {}
        self._client_db_ids = {}  # Map client_id (cid) -> database client_id
        self.metrics_dir = None
        self.experiment_name = None
        self.run_id = None
        self.training_start_time = None
        self.current_server_round = 0  # Track current server round globally

        # Initialize WebSocket sender for server-level events
        self.ws_sender = None
        if websocket_uri:
            try:
                self.ws_sender = MetricsWebSocketSender(websocket_uri)
                self.logger.info("[FederatedTrainer] WebSocket sender initialized")
            except Exception as e:
                self.logger.warning(
                    f"[FederatedTrainer] Failed to initialize WebSocket sender: {e}"
                )
                self.ws_sender = None

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

    def _create_clients_in_db(
        self,
        num_clients: int,
        db: Optional[Session] = None,
    ) -> Dict[str, int]:
        """
        Create Client records in the database for each federated client.

        Args:
            num_clients: Number of clients to create
            db: Optional database session

        Returns:
            Dictionary mapping client_id (cid) -> database client_id
        """
        close_session = False
        if db is None:
            db = get_session()
            close_session = True

        try:
            client_crud = ClientCRUD()
            client_db_ids = {}

            for client_idx in range(num_clients):
                client_identifier = str(client_idx)
                client_config = {
                    "partition_index": client_idx,
                    "client_index": client_idx,
                }

                client_record = client_crud.create_client(
                    run_id=self.run_id,
                    client_identifier=client_identifier,
                    client_config=client_config,
                )

                if client_record and hasattr(client_record, "id"):
                    client_db_ids[client_identifier] = client_record.id
                    self.logger.info(
                        f"Created Client record: client_id={client_identifier}, "
                        f"db_id={client_record.id}, run_id={self.run_id}"
                    )
                else:
                    self.logger.warning(
                        f"Failed to create Client record for client_id={client_identifier}"
                    )

            return client_db_ids

        except Exception as e:
            self.logger.error(f"Failed to create clients in database: {e}")
            raise
        finally:
            if close_session:
                db.close()

    def _create_run(
        self,
        db: Optional[Session] = None,
    ) -> int:
        """
        Create a new run in the database before training starts.

        Args:
            experiment_name: Name of the experiment
            source_path: Source data path
            db: Optional database session

        Returns:
            Created run_id
        """
        close_session = False
        if db is None:
            db = get_session()
            close_session = True

        try:
            run_data = {
                "training_mode": "federated",
                "status": "in_progress",
                "start_time": datetime.now(),
            }
            run_id = run_crud.create(db, **run_data)
            db.commit()
            self.logger.info(f"Created federated learning run with ID: {run_id}")
            return run_id.id
        except Exception as e:
            self.logger.error(f"Failed to create run: {e}")
            if close_session:
                db.rollback()
            raise
        finally:
            if close_session:
                db.close()

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

    def _client_fn(self, cid: str) -> FlowerClient:
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

        # Get database client_id for this client
        client_db_id = self._client_db_ids.get(cid)
        if not client_db_id:
            self.logger.warning(
                f"No database client ID found for client {cid}, proceeding without DB tracking"
            )

        client = FlowerClient(
            net=model,
            trainloader=train_loader,
            valloader=val_loader,
            config=self.config,
            device=self.device,
            client_id=cid,
            client_db_id=client_db_id,
            metrics_dir=self.metrics_dir,
            experiment_name=self.experiment_name,
            websocket_uri=self.websocket_uri,
            run_id=self.run_id,
            server_round=self.current_server_round,
        )

        # Store client reference for later finalization
        self._client_instances[cid] = client

        return client

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

            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
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
                    images, labels = images.to(self.device), labels.to(self.device)
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

    def train(
        self,
        data_df: pd.DataFrame,
        image_dir: Path,
        experiment_name: str = "federated_learning",
        source_path: str = "federated_data",
    ) -> Dict[str, Any]:
        """
        Execute federated learning training.

        Steps:
        1. Create run in database
        2. Send training_start WebSocket event
        3. Partition data across clients
        4. Create client dataloaders
        5. Get initial parameters
        6. Configure FedAvg strategy with evaluation
        7. Run Flower simulation
        8. Send training_end WebSocket event
        9. Return history

        Args:
            data_df: DataFrame with filename and target columns
            image_dir: Directory containing training images
            experiment_name: Name for the experiment
            source_path: Source data path for database tracking

        Returns:
            Dictionary with training results and metrics

        Raises:
            ValueError: If data_df is empty or invalid
            RuntimeError: If training fails
        """
        self.training_start_time = datetime.now()

        # Reset state for new training run
        self._client_instances = {}
        self._client_db_ids = {}
        self._client_data_cache = {}

        try:
            # Store data_df temporarily for WebSocket event setup
            self._temp_data_df = data_df

            # Step 1: Setup database run and send training_start event
            self._setup_database_run_and_events(experiment_name, source_path)

            # Step 2: Partition data and prepare dataloaders
            global_val_loader = self._partition_and_prepare_data(data_df, image_dir)

            # Step 3: Prepare model and strategy
            strategy, client_resources = self._prepare_model_and_strategy(
                global_val_loader
            )

            # Set metrics directory and experiment name for clients
            self.experiment_name = experiment_name

            # Step 4: Execute simulation , and return the results
            return self._execute_simulation(strategy, client_resources)
        except Exception as e:
            self.logger.error(f"Federated training failed: {e}")
            raise
        finally:
            self.logger.info("Federated training completed")

    def _setup_database_run_and_events(
        self,
        experiment_name: str,
        source_path: str,
    ) -> None:
        """
        Setup database run and send training start WebSocket event.

        Args:
            experiment_name: Name of the experiment
            source_path: Source data path for database tracking

        Raises:
            Exception: If run creation fails (logged but non-fatal)
        """
        self.logger.info("\nCreating run in database...")
        try:
            run_data = {
                "training_mode": "federated",
                "status": "in_progress",
                "start_time": datetime.now(),
            }
            with get_session() as db:
                self.run_id = run_crud.create(db, **run_data).id
                db.commit()
            self.logger.info(f"Run ID: {self.run_id}")
        except Exception as e:
            self.logger.warning(f"Failed to create run in database: {e}")
            self.run_id = None

        # Send training_start WebSocket event
        if self.ws_sender and self.run_id:
            try:
                self.ws_sender.send_metrics(
                    {
                        "run_id": self.run_id,
                        "experiment_name": experiment_name,
                        "training_mode": "federated",
                        "num_clients": self.config.num_clients,
                        "num_rounds": self.config.num_rounds,
                        "local_epochs": self.config.local_epochs,
                        "total_samples": len(self._temp_data_df),
                    },
                    "training_start",
                )
                self.logger.info(
                    f"[FederatedTrainer] Sent training_start event for run_id={self.run_id}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to send training_start via WebSocket: {e}")

    def _partition_and_prepare_data(
        self,
        data_df: pd.DataFrame,
        image_dir: Path,
    ) -> pd.DataFrame:
        """
        Partition data across clients and create dataloaders.

        Args:
            data_df: DataFrame with filename and target columns
            image_dir: Directory containing training images

        Returns:
            Global validation loader for server-side evaluation
        """
        self.logger.info(
            f"\nPartitioning data into {self.config.num_clients} clients..."
        )
        client_partitions = split_partition(
            partition_df=data_df,
            validation_split=self.config.validation_split,
            target_column=self.constants.TARGET_COLUMN,
            seed=self.config.seed,
        )

        # Log partition statistics
        for idx, partition in enumerate(client_partitions):
            class_dist = (
                partition[self.constants.TARGET_COLUMN].value_counts().to_dict()
            )
            self.logger.info(
                f"  Client {idx}: {len(partition)} samples, "
                f"class distribution: {class_dist}"
            )

        # Create Client records in database
        self.logger.info("\nCreating Client records in database...")
        try:
            self._client_db_ids = self._create_clients_in_db(
                num_clients=self.config.num_clients
            )
            self.logger.info(
                f"Successfully created {len(self._client_db_ids)} Client records"
            )
        except Exception as e:
            self.logger.warning(f"Failed to create Client records: {e}")
            self._client_db_ids = {}

        # Create dataloaders and cache them for client_fn
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

        # Get global validation data (reserved data not used by any client)
        self.logger.info("\nPreparing global validation data...")
        _, global_val_loader = load_data(
            partition_df=client_partitions[0],
            image_dir=image_dir,
            constants=self.constants,
            config=self.config,
        )

        return global_val_loader

    def _prepare_model_and_strategy(
        self,
        global_val_loader,
    ) -> Tuple[FedAvg, Dict]:
        """
        Prepare initial model parameters and configure FedAvg strategy.

        Args:
            global_val_loader: Validation DataLoader for global evaluation

        Returns:
            Tuple of (strategy, client_resources)
        """
        self.logger.info("\nInitializing global model parameters...")
        initial_parameters = self._get_initial_parameters()

        # Store for use in _execute_simulation
        self._global_val_loader = global_val_loader
        self._initial_parameters = initial_parameters

        self.logger.info("\nConfiguring FedAvg strategy...")
        strategy = RoundTrackingFedAvg(
            trainer_ref=self,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=self.config.num_clients,
            min_evaluate_clients=0,
            min_available_clients=self.config.num_clients,
            evaluate_fn=self._create_evaluate_fn(global_val_loader),
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=weighted_average,
        )

        # Configure client resources and metrics collection
        client_resources = {
            "num_gpus": 1.0 if self.device.type == "cuda" else 0.0,
            "num_cpus": 1.0,
        }

        # Set metrics directory and experiment name for clients
        self.metrics_dir = Path("results/federated_learning/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        return strategy, client_resources

    def _execute_simulation(
        self,
        strategy: FedAvg,
        client_resources: Dict,
    ):
        """
        Execute the Flower federated learning simulation.

        Args:
            strategy: FedAvg strategy for aggregation
            client_resources: Resource configuration for clients

        Returns:
            Training history from Flower simulation
        """
        self.logger.info("\nStarting Flower simulation...")
        self.logger.info(f"  Num rounds: {self.config.num_rounds}")
        self.logger.info(f"  Num clients: {self.config.num_clients}")
        self.logger.info(f"  Local epochs per round: {self.config.local_epochs}")
        self.logger.info("-" * 80)

        # Create custom strategy with round tracking
      
        history = fl.simulation.start_simulation(
            client_fn=self._client_fn,
            num_clients=self.config.num_clients,
            config=fl.server.ServerConfig(num_rounds=self.config.num_rounds),
            strategy=strategy,
            client_resources=client_resources,
            
        )
        results = self._finalize_training(history, self.experiment_name)
        return results

    def _finalize_training(
        self,
        history,
        experiment_name: str,
    ) -> Dict[str, Any]:
        """
        Finalize training: finalize client metrics, extract results, and send events.

        Args:
            history: Training history from Flower simulation
            experiment_name: Name of the experiment

        Returns:
            Dictionary with training results and metrics
        """
        # Finalize all client metrics
        self.logger.info("\n" + "-" * 80)
        self.logger.info("Federated Learning Completed!")
        self.logger.info("Finalizing client metrics...")
        for cid, client in self._client_instances.items():
            try:
                client.finalize()
                self.logger.info(f"  ✓ Finalized metrics for client {cid}")
            except Exception as e:
                self.logger.warning(f"  ✗ Failed to finalize client {cid}: {e}")

        # Extract results
        # Calculate best round from server evaluation metrics
        best_round = 0
        best_val_accuracy = 0.0
        best_val_loss = float("inf")

        if history.metrics_centralized:
            for round_num, (_, metrics_dict) in enumerate(
                history.metrics_centralized.get("accuracy", []), start=1
            ):
                if metrics_dict > best_val_accuracy:
                    best_val_accuracy = metrics_dict
                    best_round = round_num

            for round_num, (_, loss_value) in enumerate(
                history.losses_centralized, start=1
            ):
                if loss_value < best_val_loss:
                    best_val_loss = loss_value

        results = {
            "run_id": self.run_id,
            "experiment_name": experiment_name,
            "status": "completed",
            "num_clients": self.config.num_clients,
            "num_rounds": self.config.num_rounds,
            "local_epochs": self.config.local_epochs,
            "best_round": best_round,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "history": history,
            "metrics": {
                "losses_distributed": history.losses_distributed,
                "metrics_distributed": history.metrics_distributed,
                "metrics_centerlized": history.metrics_centralized,
                "losses_centralized": history.losses_centralized,
            },
        }

        # Calculate training duration
        training_end_time = datetime.now()
        training_duration = training_end_time - self.training_start_time
        results["training_duration"] = str(training_duration)

        # Send training_end WebSocket event
        if self.ws_sender and self.run_id:
            try:
                self.ws_sender.send_training_end(
                    run_id=self.run_id,
                    summary_data={
                        "best_round": best_round,
                        "best_val_accuracy": best_val_accuracy,
                        "best_val_loss": best_val_loss,
                        "total_rounds": self.config.num_rounds,  # num_rounds is already the total (0-indexed: 0,1,2 for 3 rounds)
                        "training_duration": str(training_duration),
                        "status": "completed",
                    },
                )
                self.logger.info(
                    f"[FederatedTrainer] Sent training_end event for run_id={self.run_id}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to send training_end via WebSocket: {e}")

        return results


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
