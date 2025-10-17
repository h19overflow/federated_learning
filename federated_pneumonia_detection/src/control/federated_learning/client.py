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

import logging
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flwr.client import NumPyClient

from federated_pneumonia_detection.src.entities.resnet_with_custom_head import (
    ResNetWithCustomHead,
)
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig


class FlowerClient(NumPyClient):
    """
    Flower NumPy client for federated learning.

    Performs local training and evaluation with injected dependencies.
    One instance per federated learning node.
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: ResNetWithCustomHead,
        device: torch.device,
        config: ExperimentConfig,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize Flower client with dependencies.

        Args:
            client_id: Unique identifier for this client
            train_loader: Training DataLoader for local data partition
            val_loader: Validation DataLoader for local data partition
            model: Pre-instantiated ResNetWithCustomHead model
            device: torch.device for training (cuda/cpu)
            config: ExperimentConfig with hyperparameters
            logger: logging.Logger instance

        Raises:
            ValueError: If any required argument is None
            TypeError: If types don't match expectations
        """
        if model is None:
            raise ValueError("model cannot be None")
        if train_loader is None:
            raise ValueError("train_loader cannot be None")
        if val_loader is None:
            raise ValueError("val_loader cannot be None")

        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.device = device
        self.config = config
        self.logger = logger

        self.logger.info(
            f"[Client {client_id}] Initialized with {len(train_loader.dataset)} "
            f"training samples and {len(val_loader.dataset)} validation samples"
        )

    def get_parameters(self, config: Dict[str, Any]) -> List:
        """Extract model parameters as numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List) -> None:
        """Load parameters from server into model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List, config: Dict[str, Any]
    ) -> Tuple[List, int, Dict[str, Any]]:
        """
        Train model on local data.

        Receives parameters from server, trains locally for local_epochs,
        returns updated parameters and training metrics.

        Args:
            parameters: Model weights from server as numpy arrays
            config: Server config with local_epochs and learning_rate

        Returns:
            Tuple of (updated_parameters, num_samples, metrics_dict)
        """
        self.logger.info(f"[Client {self.client_id}] Starting training")

        # Set global parameters
        self.set_parameters(parameters)

        # Get local training parameters
        local_epochs = config.get("local_epochs", self.config.local_epochs)
        learning_rate = config.get("lr", self.config.learning_rate)

        # Create optimizer and loss function
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay
        )

        criterion = nn.BCEWithLogitsLoss() if self.config.num_classes == 1 else nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        total_loss = 0.0
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)

                if self.config.num_classes == 1:
                    labels = labels.float().unsqueeze(1)
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels.long())

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            total_loss += epoch_loss / len(self.train_loader)

        avg_loss = total_loss / local_epochs if local_epochs > 0 else 0.0

        # Return updated parameters
        num_samples = len(self.train_loader.dataset)
        metrics = {"train_loss": avg_loss}

        self.logger.info(
            f"[Client {self.client_id}] Training complete: "
            f"loss={avg_loss:.4f}, samples={num_samples}"
        )

        return self.get_parameters(config={}), num_samples, metrics

    def evaluate(
        self, parameters: List, config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model on local validation data.

        Receives parameters from server, evaluates on local validation set,
        returns loss and metrics.

        Args:
            parameters: Model weights from server as numpy arrays
            config: Server evaluation config (unused)

        Returns:
            Tuple of (loss, num_samples, metrics_dict with accuracy)
        """
        self.logger.info(f"[Client {self.client_id}] Starting evaluation")

        # Set global parameters
        self.set_parameters(parameters)

        # Evaluate
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss() if self.config.num_classes == 1 else nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                if self.config.num_classes == 1:
                    labels_for_loss = labels.float().unsqueeze(1)
                    loss = criterion(outputs, labels_for_loss)
                    predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()
                    correct += (predictions == labels).sum().item()
                else:
                    loss = criterion(outputs, labels.long())
                    _, predictions = torch.max(outputs, 1)
                    correct += (predictions == labels).sum().item()

                total_loss += loss.item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0

        num_samples = len(self.val_loader.dataset)
        metrics = {"accuracy": accuracy}

        self.logger.info(
            f"[Client {self.client_id}] Evaluation complete: "
            f"loss={avg_loss:.4f}, accuracy={accuracy:.4f}, samples={num_samples}"
        )

        return avg_loss, num_samples, metrics
