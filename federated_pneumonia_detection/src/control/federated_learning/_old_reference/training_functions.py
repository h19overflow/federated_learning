"""
Pure PyTorch training and evaluation functions for federated learning.
Provides training loops without PyTorch Lightning for Flower compatibility.
"""

import logging
from typing import Tuple, Dict, List, Optional
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from federated_pneumonia_detection.src.entities.resnet_with_custom_head import ResNetWithCustomHead


def train_one_epoch(
    model: ResNetWithCustomHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 1,
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Train model for one epoch using standard PyTorch.

    Args:
        model: ResNetWithCustomHead model to train
        dataloader: Training DataLoader
        optimizer: Optimizer instance
        device: Device to train on
        num_classes: Number of classes (1 for binary classification)
        logger: Optional logger instance

    Returns:
        Average training loss for the epoch
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    model.train()
    model.to(device)

    # Setup loss function
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate loss
        if num_classes == 1:
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
        else:
            labels = labels.long()
            loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    logger.info(f"Training completed: avg_loss={avg_loss:.4f}")

    return avg_loss


def evaluate_model(
    model: ResNetWithCustomHead,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 1,
    logger: Optional[logging.Logger] = None
) -> Tuple[float, float, Dict[str, float]]:
    """
    Evaluate model on validation/test data.

    Args:
        model: Model to evaluate
        dataloader: Validation/test DataLoader
        device: Device to evaluate on
        num_classes: Number of classes
        logger: Optional logger instance

    Returns:
        Tuple of (loss, accuracy, metrics_dict)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    model.eval()
    model.to(device)

    # Setup loss function
    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    # For additional metrics
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            if num_classes == 1:
                labels_for_loss = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels_for_loss)

                # Get predictions
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions.squeeze() == labels).sum().item()
            else:
                loss = criterion(outputs, labels.long())

                # Get predictions
                _, predictions = torch.max(outputs, 1)
                correct += (predictions == labels).sum().item()

            total_loss += loss.item()
            total += labels.size(0)

            # Store for metrics calculation
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    # Calculate additional metrics
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'num_samples': total
    }

    logger.info(f"Evaluation: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

    return avg_loss, accuracy, metrics


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as list of numpy arrays for Flower.

    Args:
        model: PyTorch model

    Returns:
        List of numpy arrays containing model weights
    """
    return [param.cpu().detach().numpy() for param in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load parameters from numpy arrays into PyTorch model.

    Args:
        model: PyTorch model to update
        parameters: List of numpy arrays containing weights
    """
    params_dict = zip(model.parameters(), parameters)
    for param, new_param in params_dict:
        param.data = torch.from_numpy(new_param).to(param.device)


def get_model_state_dict(model: nn.Module) -> OrderedDict:
    """
    Get model state dictionary for checkpointing.

    Args:
        model: PyTorch model

    Returns:
        Ordered dictionary with model state
    """
    return model.state_dict()


def load_model_state_dict(model: nn.Module, state_dict: OrderedDict) -> None:
    """
    Load state dictionary into model.

    Args:
        model: PyTorch model
        state_dict: State dictionary to load
    """
    model.load_state_dict(state_dict)


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0001
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer for model.

    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization

    Returns:
        AdamW optimizer instance
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )


def train_multiple_epochs(
    model: ResNetWithCustomHead,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    num_classes: int = 1,
    weight_decay: float = 0.0001,
    logger: Optional[logging.Logger] = None
) -> Dict[str, List[float]]:
    """
    Train model for multiple epochs with validation.

    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (optional)
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to train on
        num_classes: Number of classes
        weight_decay: Weight decay for optimizer
        logger: Optional logger instance

    Returns:
        Dictionary with training history (losses, accuracies)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    optimizer = create_optimizer(model, learning_rate, weight_decay)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Training
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, num_classes, logger
        )
        history['train_loss'].append(train_loss)

        # Validation
        if val_loader is not None:
            val_loss, val_acc, _ = evaluate_model(
                model, val_loader, device, num_classes, logger
            )
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            logger.info(
                f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )
        else:
            logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}")

    return history
