"""
Federated learning system for pneumonia detection.

Simplified architecture following modern Flower API patterns.
"""

from .client import FlowerClient, _evaluate, _train, _get_weights, _set_weights
from .data_manager import load_data, split_partition
from .trainer import FederatedTrainer

__all__ = [
    "FlowerClient",
    "train",
    "_evaluate",
    "_train",
    "_get_weights",
    "_set_weights",
    "load_data",
    "split_partition",
    "FederatedTrainer",
]
