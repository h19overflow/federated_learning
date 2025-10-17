"""
Federated learning system for pneumonia detection.

Simplified architecture following modern Flower API patterns.
"""

from .client import FlowerClient, train, evaluate, get_weights, set_weights
from .data_manager import load_data, split_partition
from .partitioner import partition_data_stratified
from .trainer import FederatedTrainer

__all__ = [
    'FlowerClient',
    'train',
    'evaluate',
    'get_weights',
    'set_weights',
    'load_data',
    'split_partition',
    'partition_data_stratified',
    'FederatedTrainer',
]
