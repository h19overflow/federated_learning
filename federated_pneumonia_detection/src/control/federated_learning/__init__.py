"""
Federated learning system for pneumonia detection.

Simplified architecture following modern Flower API patterns.
"""

from .trainer import FederatedTrainer
from .client import FlowerClient

__all__ = ['FederatedTrainer', 'FlowerClient']
