"""DataLoader configuration utilities for XRayDataModule."""

from typing import TYPE_CHECKING
import torch
import numpy as np
from federated_pneumonia_detection.src.utils.loggers.logger import get_logger

if TYPE_CHECKING:
    from federated_pneumonia_detection.config.config_manager import ConfigManager

logger = get_logger(__name__)


def build_dataloader_kwargs(
    config: 'ConfigManager',
    shuffle: bool,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> dict:
    """
    Build DataLoader configuration dictionary.

    Args:
        config: ConfigManager instance
        shuffle: Whether to shuffle data
        pin_memory: Whether to use pinned memory
        persistent_workers: Whether to keep workers alive
        prefetch_factor: Number of batches to prefetch

    Returns:
        Dictionary with DataLoader kwargs
    """
    num_workers = config.get('experiment.num_workers', 4)

    loader_kwargs = {
        'batch_size': config.get('experiment.batch_size', 32),
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': False,
    }

    # Only set these parameters when using multiprocessing (num_workers > 0)
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['worker_init_fn'] = worker_init_fn

    return loader_kwargs


def worker_init_fn(worker_id: int):
    """
    Initialize worker processes with different random seeds.

    Args:
        worker_id: Worker process ID

    Note:
        This prevents duplicate augmentations across workers by setting
        unique random seeds for each worker process.
    """
    # Set different seed for each worker to avoid duplicate augmentations
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    torch.manual_seed(worker_seed + worker_id)

    logger.debug(f"Worker {worker_id} initialized with seed {worker_seed + worker_id}")
