"""
Data partitioning utility for federated learning.

Splits datasets across multiple clients using stratified sampling to maintain
class distribution. Ensures IID (Independent and Identically Distributed) data
partitioning across federated clients for fair model training.

Dependencies:
- pandas: DataFrame operations
- numpy: Array splitting and shuffling
- logging: Optional logging support

Role in System:
- Distributes data fairly across federated clients
- Maintains class balance via stratified sampling
- Enables repeatable partitioning through seeding
"""

import logging
from typing import List, Optional
import numpy as np
import pandas as pd


def partition_data_stratified(
    df: pd.DataFrame,
    num_clients: int,
    target_column: str,
    seed: int = 42,
    logger: Optional[logging.Logger] = None
) -> List[pd.DataFrame]:
    """
    Partition data across clients while maintaining class distribution.

    Splits each class independently and distributes portions to each client,
    ensuring balanced class representation across all partitions. This creates
    IID-partitioned data suitable for federated learning.

    Args:
        df: DataFrame to partition
        num_clients: Number of federated clients (must be > 0)
        target_column: Column name containing class labels
        seed: Random seed for reproducible shuffling (default: 42)
        logger: Optional logger for partition size reporting

    Returns:
        List of DataFrames, one per client, each with balanced class distribution

    Raises:
        ValueError: If num_clients <= 0 or target_column not in DataFrame
        TypeError: If df is not a pandas DataFrame

    Example:
        >>> df = pd.DataFrame({
        ...     'image_id': [1, 2, 3, 4, 5, 6],
        ...     'label': ['normal', 'pneumonia', 'normal', 'pneumonia', 'normal', 'pneumonia']
        ... })
        >>> partitions = partition_data_stratified(df, num_clients=3, target_column='label')
        >>> len(partitions)
        3
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if num_clients <= 0:
        raise ValueError(f"num_clients must be > 0, got {num_clients}")

    if target_column not in df.columns:
        raise ValueError(
            f"target_column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    np.random.seed(seed)

    # Initialize client partitions
    client_partitions: List[pd.DataFrame] = [pd.DataFrame() for _ in range(num_clients)]

    # Distribute each class across clients
    for class_label in df[target_column].unique():
        class_data = df[df[target_column] == class_label].copy()

        # Split class data into num_clients parts
        class_splits = np.array_split(class_data, num_clients)

        # Assign each split to corresponding client
        for client_idx, split in enumerate(class_splits):
            client_partitions[client_idx] = pd.concat(
                [client_partitions[client_idx], split],
                ignore_index=True
            )

    # Shuffle each partition
    for i in range(num_clients):
        client_partitions[i] = client_partitions[i].sample(
            frac=1,
            random_state=seed + i
        ).reset_index(drop=True)

    # Log partition information
    if logger:
        logger.info(f"Data partitioned across {num_clients} clients")
        for i, partition in enumerate(client_partitions):
            class_dist = partition[target_column].value_counts().to_dict()
            logger.debug(
                f"Client {i}: {len(partition)} samples, "
                f"class distribution: {class_dist}"
            )

    return client_partitions
