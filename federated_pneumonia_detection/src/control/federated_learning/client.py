"""
Federated learning client for local model training.

This module implements a Flower NumPyClient that performs local training
and evaluation on a client's data partition following the modern Flower API.

Dependencies:
- flwr.client.NumPyClient: Base class for Flower clients
- PyTorch: For model training and evaluation

Role in System:
- Receives global model parameters from server
- Trains on local data partition
- Evaluates on local validation data
- Returns updated parameters and metrics to server
"""

from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flwr.client import NumPyClient

from federated_pneumonia_detection.src.entities.resnet_with_custom_head import (
    ResNetWithCustomHead,
)
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.control.federated_learning.federated_metrics_collector import (
    FederatedMetricsCollector,
)
from federated_pneumonia_detection.src.control.dl_model.utils.data.websocket_metrics_sender import MetricsWebSocketSender

# TODO: Complete the logging functionality throughout the client, as well as collecting the results.
class FlowerClient(NumPyClient):
    """Flower NumPy client for federated learning."""

    def __init__(
        self,
        net: ResNetWithCustomHead,
        trainloader: DataLoader,
        valloader: DataLoader,
        config: ExperimentConfig,
        device: torch.device,
        client_id: Optional[str] = None,
        metrics_dir: Optional[str] = None,
        experiment_name: str = "federated_pneumonia",
        websocket_uri: Optional[str] = "ws://localhost:8765",
        run_id: Optional[int] = None,
    ) -> None:
        """
        Initialize Flower client with dependencies.

        Args:
            net: Neural network model
            trainloader: Training data loader
            valloader: Validation data loader
            config: Experiment configuration
            device: Device for training (CPU/GPU)
            client_id: Unique identifier for this client
            metrics_dir: Directory to save metrics (None = no metrics collection)
            experiment_name: Name of the experiment
            websocket_uri: WebSocket URI for real-time metrics streaming
            run_id: Optional database run ID for persistence
        """
        if net is None:
            raise ValueError("net cannot be None")
        if trainloader is None:
            raise ValueError("trainloader cannot be None")
        if valloader is None:
            raise ValueError("valloader cannot be None")

        # Validate that client has sufficient data
        train_samples = (
            len(trainloader.dataset) if hasattr(trainloader, "dataset") else 0
        )
        if train_samples == 0:
            raise ValueError(
                f"Client {client_id or 'unknown'} has no training samples. "
                "Check data partitioning and sample_fraction settings."
            )

        # Warn if batch size is larger than dataset
        if (
            hasattr(trainloader, "batch_size")
            and trainloader.batch_size
            and train_samples < trainloader.batch_size
        ):
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Client {client_id or 'unknown'}: batch_size ({trainloader.batch_size}) "
                f"> train_samples ({train_samples}). This will result in only 1 batch per epoch."
            )

        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.config = config
        self.device = device
        self.client_id = client_id or "client_0"
        self.current_round = 0
        self.websocket_uri = websocket_uri
        self.run_id = run_id
        self.experiment_name = experiment_name

        # Initialize WebSocket sender for direct client communication
        self.ws_sender = None
        if websocket_uri:
            try:
                self.ws_sender = MetricsWebSocketSender(websocket_uri)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to initialize WebSocket sender for client {self.client_id}: {e}")

        # Initialize metrics collector if directory provided
        self.metrics_collector = None
        if metrics_dir:
            self.metrics_collector = FederatedMetricsCollector(
                save_dir=metrics_dir,
                client_id=self.client_id,
                experiment_name=experiment_name,
                run_id=run_id,
                enable_progress_logging=True,
                websocket_uri=websocket_uri,
            )

            # Record model info
            model_info = {
                "total_parameters": sum(p.numel() for p in net.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in net.parameters() if p.requires_grad
                ),
                "train_samples": len(trainloader.dataset),
                "val_samples": len(valloader.dataset),
                "device": str(device),
            }
            self.metrics_collector.start_training(model_info)

    def get_parameters(self, config: Dict[str, Any]) -> List:
        """Extract model parameters as numpy arrays."""
        return get_weights(self.net)

    def set_parameters(self, parameters: List) -> None:
        """Load parameters from server into model."""
        set_weights(self.net, parameters)

    def fit(
        self, parameters: List, config: Dict[str, Any]
    ) -> Tuple[List, int, Dict[str, Any]]:
        """Train model on local data."""
        self.set_parameters(parameters)

        local_epochs = config.get("local_epochs", self.config.local_epochs)
        learning_rate = config.get("lr", self.config.learning_rate)

        # Send WebSocket notification that this client is starting training
        if self.ws_sender:
            try:
                self.ws_sender.send_metrics({
                    "run_id": self.run_id,
                    "client_id": self.client_id,
                    "round": self.current_round,
                    "status": "client_training_started",
                    "local_epochs": local_epochs,
                    "num_samples": len(self.trainloader.dataset),
                    "experiment_name": self.experiment_name
                }, "client_training_start")
            except Exception as e:
                pass  # Silently fail if WebSocket unavailable

        # Record round start if metrics collector is active
        if self.metrics_collector:
            self.metrics_collector.record_round_start(
                round_num=self.current_round, server_config=config
            )

        train_loss, epoch_losses = train(
            self.net,
            self.trainloader,
            local_epochs,
            self.device,
            learning_rate,
            self.config.weight_decay,
            self.config.num_classes,
            metrics_collector=self.metrics_collector,
            current_round=self.current_round,
        )

        # Record fit metrics
        if self.metrics_collector:
            self.metrics_collector.record_fit_metrics(
                round_num=self.current_round,
                train_loss=train_loss,
                num_samples=len(self.trainloader.dataset),
            )

        # Send WebSocket notification that this client finished training
        if self.ws_sender:
            try:
                self.ws_sender.send_metrics({
                    "run_id": self.run_id,
                    "client_id": self.client_id,
                    "round": self.current_round,
                    "status": "client_training_completed",
                    "train_loss": train_loss,
                    "num_samples": len(self.trainloader.dataset),
                    "experiment_name": self.experiment_name
                }, "client_training_end")
            except Exception as e:
                pass  # Silently fail if WebSocket unavailable

        self.current_round += 1

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(
        self, parameters: List, config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate model on local validation data."""
        self.set_parameters(parameters)

        loss, accuracy = evaluate(
            self.net,
            self.valloader,
            self.device,
            self.config.num_classes,
        )

        # Record evaluation metrics
        # Note: current_round - 1 because fit() already incremented it
        if self.metrics_collector:
            self.metrics_collector.record_eval_metrics(
                round_num=self.current_round - 1,
                val_loss=loss,
                val_accuracy=accuracy,
                num_samples=len(self.valloader.dataset),
            )

        # Send WebSocket notification that this client finished evaluation
        if self.ws_sender:
            try:
                self.ws_sender.send_metrics({
                    "run_id": self.run_id,
                    "client_id": self.client_id,
                    "round": self.current_round - 1,
                    "status": "client_evaluation_completed",
                    "val_loss": loss,
                    "val_accuracy": accuracy,
                    "num_samples": len(self.valloader.dataset),
                    "experiment_name": self.experiment_name
                }, "client_eval_end")
            except Exception as e:
                pass  # Silently fail if WebSocket unavailable

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

    def finalize(self):
        """Finalize training and save all collected metrics."""
        if self.metrics_collector:
            self.metrics_collector.end_training()


# Helper functions


def get_weights(net: ResNetWithCustomHead) -> List:
    """Extract model weights as numpy arrays."""
    return [val.cpu().numpy() for val in net.state_dict().values()]


def set_weights(net: ResNetWithCustomHead, parameters: List) -> None:
    """Load weights into model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def  train(
    net: ResNetWithCustomHead,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    num_classes: int,
    metrics_collector: Optional[FederatedMetricsCollector] = None,
    current_round: int = 0,
) -> Tuple[float, List[float]]:
    """
    Train the model on the training set.

    Args:
        net: Neural network model
        trainloader: Training data loader
        epochs: Number of local epochs
        device: Device for training
        learning_rate: Learning rate
        weight_decay: Weight decay
        num_classes: Number of output classes
        metrics_collector: Optional metrics collector
        current_round: Current federated round

    Returns:
        Tuple of (average_loss, epoch_losses)
    """
    net.to(device)
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    net.train()

    epoch_losses = []
    total_running_loss = 0.0
    total_batches_all_epochs = len(trainloader) * epochs
    processed_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(images)

            if num_classes == 1:
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            processed_batches += 1

        # Calculate average loss for this epoch
        # Guard against division by zero when no batches were processed
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            total_running_loss += epoch_loss
        else:
            # No data to train on - append 0.0 as placeholder
            epoch_losses.append(0.0)

        # Record epoch metrics if collector is available
        if metrics_collector and num_batches > 0:
            progress_percent = (processed_batches / total_batches_all_epochs * 100) if total_batches_all_epochs > 0 else 0
            metrics_collector.record_local_epoch(
                round_num=current_round,
                local_epoch=epoch,
                train_loss=avg_epoch_loss,
                learning_rate=learning_rate,
                num_samples=len(trainloader.dataset),
                additional_metrics={
                    "batch_count": num_batches,
                    "epoch_progress": f"{epoch + 1}/{epochs}",
                    "overall_progress_percent": progress_percent,
                    "batches_processed": processed_batches,
                    "total_batches": total_batches_all_epochs,
                }
            )

    # Calculate final average, guarding against empty trainloader
    total_batches = len(trainloader) * epochs
    avg_trainloss = total_running_loss / total_batches if total_batches > 0 else 0.0
    return avg_trainloss, epoch_losses


def evaluate(
    net: ResNetWithCustomHead,
    valloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float]:
    """Evaluate the model on the validation set."""
    net.to(device)
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)

            if num_classes == 1:
                labels_for_loss = labels.float().unsqueeze(1)
                loss += criterion(outputs, labels_for_loss).item()
                predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                correct += (predictions == labels).sum().item()
            else:
                loss += criterion(outputs, labels.long()).item()
                _, predictions = torch.max(outputs, 1)
                correct += (predictions == labels).sum().item()

    accuracy = correct / len(valloader.dataset)
    loss = loss / len(valloader)
    return loss, accuracy
