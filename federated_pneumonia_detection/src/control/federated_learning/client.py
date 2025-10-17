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
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flwr.client import NumPyClient

from federated_pneumonia_detection.src.entities.resnet_with_custom_head import (
    ResNetWithCustomHead,
)
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig


def get_weights(net: ResNetWithCustomHead) -> List:
    """Extract model weights as numpy arrays."""
    return [val.cpu().numpy() for val in net.state_dict().values()]


def set_weights(net: ResNetWithCustomHead, parameters: List) -> None:
    """Load weights into model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(
    net: ResNetWithCustomHead,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    num_classes: int,
) -> float:
    """Train the model on the training set."""
    net.to(device)
    criterion = (
        nn.BCEWithLogitsLoss()
        if num_classes == 1
        else nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
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
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def evaluate(
    net: ResNetWithCustomHead,
    valloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float]:
    """Evaluate the model on the validation set."""
    net.to(device)
    criterion = (
        nn.BCEWithLogitsLoss()
        if num_classes == 1
        else nn.CrossEntropyLoss()
    )
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


class FlowerClient(NumPyClient):
    """Flower NumPy client for federated learning."""

    def __init__(
        self,
        net: ResNetWithCustomHead,
        trainloader: DataLoader,
        valloader: DataLoader,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        """Initialize Flower client with dependencies."""
        if net is None:
            raise ValueError("net cannot be None")
        if trainloader is None:
            raise ValueError("trainloader cannot be None")
        if valloader is None:
            raise ValueError("valloader cannot be None")

        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.config = config
        self.device = device

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

        train_loss = train(
            self.net,
            self.trainloader,
            local_epochs,
            self.device,
            learning_rate,
            self.config.weight_decay,
            self.config.num_classes,
        )

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

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}
