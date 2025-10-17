"""
Usage example for ClientDataManager.

Demonstrates how to create DataLoaders for federated learning clients
from data partitions.
"""

import logging
import pandas as pd
from pathlib import Path

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.control.federated_learning.data_manager import ClientDataManager
from federated_pneumonia_detection.src.utils.logger import get_logger


def example_basic_usage():
    """Basic example: Create DataLoaders for a client partition."""

    # Initialize logging
    logger = get_logger(__name__)

    # Setup configuration
    constants = SystemConstants()
    config = ExperimentConfig(
        batch_size=64,
        validation_split=0.2,
        augmentation_strength=1.0,
        use_custom_preprocessing=False,
        pin_memory=True,
        color_mode='RGB'
    )

    # Initialize data manager
    image_dir = Path('./data/Images')  # Path to client's image directory
    data_manager = ClientDataManager(
        image_dir=image_dir,
        constants=constants,
        config=config,
        logger=logger
    )

    # Create sample partition DataFrame
    partition_df = pd.DataFrame({
        'filename': [
            'patient001_image.png',
            'patient002_image.png',
            'patient003_image.png',
            'patient004_image.png',
            'patient005_image.png',
        ],
        'Target': [0, 1, 0, 1, 0]  # 0=Normal, 1=Pneumonia
    })

    # Create DataLoaders with default validation split
    train_loader, val_loader = data_manager.create_dataloaders_for_partition(
        partition_df=partition_df
    )

    # Use DataLoaders
    logger.info(f"Train loader batches: {len(train_loader)}")
    logger.info(f"Val loader batches: {len(val_loader)}")

    # Iterate through training batches
    for batch_idx, (images, labels) in enumerate(train_loader):
        logger.info(f"Batch {batch_idx}: images shape={images.shape}, labels shape={labels.shape}")
        if batch_idx == 0:  # Show only first batch
            break


def example_custom_validation_split():
    """Example: Use custom validation split percentage."""

    logger = get_logger(__name__)

    constants = SystemConstants()
    config = ExperimentConfig(batch_size=32)

    data_manager = ClientDataManager(
        image_dir=Path('./data/Images'),
        constants=constants,
        config=config,
        logger=logger
    )

    # Create sample partition
    partition_df = pd.DataFrame({
        'filename': [f'image_{i}.png' for i in range(100)],
        'Target': [i % 2 for i in range(100)]
    })

    # Override validation split to 30%
    train_loader, val_loader = data_manager.create_dataloaders_for_partition(
        partition_df=partition_df,
        validation_split=0.3
    )

    logger.info(f"With 30% validation split:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")


def example_with_augmentation():
    """Example: Enable augmentation for training."""

    logger = get_logger(__name__)

    constants = SystemConstants(IMG_SIZE=(256, 256))
    config = ExperimentConfig(
        batch_size=64,
        augmentation_strength=1.5,  # Increase augmentation
        validation_split=0.2,
        use_custom_preprocessing=False
    )

    data_manager = ClientDataManager(
        image_dir=Path('./data/Images'),
        constants=constants,
        config=config,
        logger=logger
    )

    partition_df = pd.DataFrame({
        'filename': [f'image_{i}.png' for i in range(50)],
        'Target': [i % 2 for i in range(50)]
    })

    train_loader, val_loader = data_manager.create_dataloaders_for_partition(
        partition_df=partition_df
    )

    logger.info("DataLoaders created with augmentation enabled")


def example_federated_client_simulation():
    """
    Example: Simulate federated learning client setup.

    Shows how a Flower client would use ClientDataManager
    to load its assigned data partition.
    """

    logger = get_logger(__name__)

    # Client-specific configuration
    constants = SystemConstants()
    config = ExperimentConfig(
        batch_size=128,
        validation_split=0.2,
        augmentation_strength=1.0,
        pin_memory=False,  # Disable on some systems
        color_mode='RGB'
    )

    # Client's data directory
    client_id = 'client_001'
    client_image_dir = Path(f'./federated_data/{client_id}/images')

    # Initialize manager
    try:
        data_manager = ClientDataManager(
            image_dir=client_image_dir,
            constants=constants,
            config=config,
            logger=logger
        )
    except ValueError as e:
        logger.error(f"Failed to initialize data manager: {e}")
        return

    # Simulate client partition (would come from data distribution)
    client_partition = pd.DataFrame({
        'filename': [f'{client_id}_image_{i}.png' for i in range(75)],
        'Target': [i % 2 for i in range(75)]
    })

    try:
        train_loader, val_loader = data_manager.create_dataloaders_for_partition(
            partition_df=client_partition
        )
        logger.info(f"Client {client_id} ready for training")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
    except RuntimeError as e:
        logger.error(f"Failed to create DataLoaders: {e}")


def example_error_handling():
    """Example: Demonstrate error handling."""

    logger = get_logger(__name__)

    constants = SystemConstants()
    config = ExperimentConfig(batch_size=64)

    # This will raise ValueError because image_dir doesn't exist
    try:
        data_manager = ClientDataManager(
            image_dir=Path('/nonexistent/path'),
            constants=constants,
            config=config,
            logger=logger
        )
    except ValueError as e:
        logger.info(f"Caught expected error: {e}")

    # Create valid manager
    data_manager = ClientDataManager(
        image_dir=Path('./data/Images'),
        constants=constants,
        config=config,
        logger=logger
    )

    # Empty DataFrame will raise ValueError
    try:
        empty_df = pd.DataFrame({'filename': [], 'Target': []})
        data_manager.create_dataloaders_for_partition(empty_df)
    except ValueError as e:
        logger.info(f"Caught expected error: {e}")

    # Missing required columns will raise ValueError
    try:
        bad_df = pd.DataFrame({'wrong_col': ['file1.png']})
        data_manager.create_dataloaders_for_partition(bad_df)
    except ValueError as e:
        logger.info(f"Caught expected error: {e}")


if __name__ == '__main__':
    # Uncomment examples to run

    # example_basic_usage()
    # example_custom_validation_split()
    # example_with_augmentation()
    # example_federated_client_simulation()
    example_error_handling()
