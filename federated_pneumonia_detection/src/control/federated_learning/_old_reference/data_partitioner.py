"""
Data partitioning utilities for federated learning.
Splits datasets across multiple clients using IID or non-IID strategies.
"""

import logging
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.src.entities.custom_image_dataset import CustomImageDataset
from federated_pneumonia_detection.src.utils.image_transforms import TransformBuilder


def partition_data_iid(
    df: pd.DataFrame,
    num_clients: int,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> List[pd.DataFrame]:
    """
    Partition data randomly across clients (IID - Independent and Identically Distributed).

    Args:
        df: DataFrame to partition
        num_clients: Number of clients to partition data across
        seed: Random seed for reproducibility
        logger: Optional logger instance

    Returns:
        List of DataFrames, one per client

    Raises:
        ValueError: If num_clients is invalid or partitioning fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    if len(df) < num_clients:
        raise ValueError(f"Dataset size ({len(df)}) smaller than num_clients ({num_clients})")

    # Shuffle data with fixed seed
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Split into roughly equal partitions
    partitions = np.array_split(df_shuffled, num_clients)

    logger.info(f"IID partitioning: {num_clients} clients, sizes: {[len(p) for p in partitions]}")

    return partitions


def partition_data_by_patient(
    df: pd.DataFrame,
    num_clients: int,
    patient_column: str,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> List[pd.DataFrame]:
    """
    Partition data by patient IDs (non-IID, more realistic for medical data).
    Each client gets data from distinct set of patients.

    Args:
        df: DataFrame to partition
        num_clients: Number of clients
        patient_column: Column name containing patient IDs
        seed: Random seed for reproducibility
        logger: Optional logger instance

    Returns:
        List of DataFrames, one per client

    Raises:
        ValueError: If partitioning fails or parameters are invalid
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    if patient_column not in df.columns:
        raise ValueError(f"Column '{patient_column}' not found in DataFrame")

    # Get unique patients and shuffle
    unique_patients = df[patient_column].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_patients)

    if len(unique_patients) < num_clients:
        logger.warning(
            f"Fewer patients ({len(unique_patients)}) than clients ({num_clients}). "
            "Some clients will have no data."
        )

    # Partition patients across clients
    patient_partitions = np.array_split(unique_patients, num_clients)

    # Create DataFrames for each client
    partitions = []
    for i, patient_ids in enumerate(patient_partitions):
        client_df = df[df[patient_column].isin(patient_ids)].reset_index(drop=True)
        partitions.append(client_df)

        logger.info(f"Client {i}: {len(client_df)} samples from {len(patient_ids)} patients")

    return partitions


def partition_data_stratified(
    df: pd.DataFrame,
    num_clients: int,
    target_column: str,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> List[pd.DataFrame]:
    """
    Partition data while maintaining class distribution across clients (stratified IID).

    Args:
        df: DataFrame to partition
        num_clients: Number of clients
        target_column: Column containing class labels
        seed: Random seed for reproducibility
        logger: Optional logger instance

    Returns:
        List of DataFrames with balanced class distributions

    Raises:
        ValueError: If partitioning fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame")

    # Initialize empty partitions
    partitions = [pd.DataFrame() for _ in range(num_clients)]

    # For each class, split samples across clients
    for class_label in df[target_column].unique():
        class_df = df[df[target_column] == class_label].sample(frac=1.0, random_state=seed)
        class_partitions = np.array_split(class_df, num_clients)

        # Append to each client's partition
        for i, class_partition in enumerate(class_partitions):
            partitions[i] = pd.concat([partitions[i], class_partition], ignore_index=True)

    # Shuffle each partition
    partitions = [p.sample(frac=1.0, random_state=seed).reset_index(drop=True) for p in partitions]

    logger.info(f"Stratified partitioning: {num_clients} clients, sizes: {[len(p) for p in partitions]}")

    return partitions


def create_client_dataloaders(
    partitions: List[pd.DataFrame],
    image_dir: str,
    constants: SystemConstants,
    config: ExperimentConfig,
    is_training: bool = True,
    logger: Optional[logging.Logger] = None
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Create DataLoaders for each client partition with train/val split.

    Args:
        partitions: List of DataFrames, one per client
        image_dir: Directory containing images
        constants: SystemConstants for configuration
        config: ExperimentConfig for parameters
        is_training: Whether to create training or validation dataloaders
        logger: Optional logger instance

    Returns:
        List of (train_loader, val_loader) tuples, one per client
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    transform_builder = TransformBuilder(constants, config)
    client_dataloaders = []

    for i, partition_df in enumerate(partitions):
        if len(partition_df) == 0:
            logger.warning(f"Client {i} has empty partition, skipping")
            continue

        # Split into train/val for this client
        from federated_pneumonia_detection.src.utils.data_processing import create_train_val_split

        try:
            train_df, val_df = create_train_val_split(
                partition_df,
                validation_split=config.validation_split,
                target_column=constants.TARGET_COLUMN,
                seed=config.seed,
                logger=logger
            )

            # Create transforms
            train_transforms = transform_builder.build_training_transforms(
                enable_augmentation=is_training,
                augmentation_strength=config.augmentation_strength
            )
            val_transforms = transform_builder.build_validation_transforms()

            # Create datasets
            train_dataset = CustomImageDataset(
                dataframe=train_df,
                image_dir=image_dir,
                constants=constants,
                transform=train_transforms,
                color_mode=config.color_mode,
                validate_images=False
            )

            val_dataset = CustomImageDataset(
                dataframe=val_df,
                image_dir=image_dir,
                constants=constants,
                transform=val_transforms,
                color_mode=config.color_mode,
                validate_images=False
            )

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                drop_last=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                drop_last=False
            )

            client_dataloaders.append((train_loader, val_loader))

            logger.info(
                f"Client {i} dataloaders created: "
                f"{len(train_dataset)} train, {len(val_dataset)} val samples"
            )

        except Exception as e:
            logger.error(f"Failed to create dataloaders for client {i}: {e}")
            raise

    return client_dataloaders
