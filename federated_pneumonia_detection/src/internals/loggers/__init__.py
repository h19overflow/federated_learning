"""
Logging utilities for training observability.

Provides multiprocessing-safe loggers for centralized and federated training.
"""

from .logger import setup_logger

__all__ = [
    'FederatedProgressLogger',
    'setup_logger'
]
