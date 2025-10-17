"""
Usage examples and integration tests for the partitioner utility.

This module demonstrates how to use the partition_data_stratified function
in federated learning scenarios.
"""

import logging
import pandas as pd
import numpy as np

from federated_pneumonia_detection.src.control.federated_learning.partitioner import (
    partition_data_stratified,
)


def example_basic_usage():
    """Example: Basic data partitioning with balanced classes."""
    print("\n=== Example 1: Basic Partitioning ===")

    # Create sample data with balanced classes
    df = pd.DataFrame({
        'image_id': list(range(1, 13)),
        'label': ['normal', 'pneumonia'] * 6,
        'filename': [f'image_{i}.jpg' for i in range(1, 13)]
    })

    print(f"Original dataset: {len(df)} samples")
    print(f"  Class distribution: {df['label'].value_counts().to_dict()}")

    # Partition across 3 clients
    partitions = partition_data_stratified(
        df,
        num_clients=3,
        target_column='label',
        seed=42
    )

    print(f"\nPartitioned into {len(partitions)} client datasets:")
    for i, partition in enumerate(partitions):
        print(f"  Client {i}: {len(partition)} samples")
        print(f"    Classes: {partition['label'].value_counts().to_dict()}")


def example_with_logging():
    """Example: Partitioning with logging for debugging."""
    print("\n=== Example 2: With Logging ===")

    # Setup logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Create imbalanced dataset (common in real scenarios)
    df = pd.DataFrame({
        'image_id': list(range(1, 21)),
        'label': ['normal'] * 14 + ['pneumonia'] * 6,
    })

    print(f"Original dataset: {len(df)} samples")
    print(f"  Class distribution: {df['label'].value_counts().to_dict()}")

    # Partition with logging
    partitions = partition_data_stratified(
        df,
        num_clients=4,
        target_column='label',
        seed=42,
        logger=logger
    )

    print(f"\nPartitioned into {len(partitions)} clients (check logs above)")


def example_reproducibility():
    """Example: Demonstrating reproducible partitioning with seed."""
    print("\n=== Example 3: Reproducibility with Seed ===")

    df = pd.DataFrame({
        'image_id': list(range(1, 9)),
        'label': ['normal', 'pneumonia'] * 4,
    })

    # Partition with seed=42
    partitions1 = partition_data_stratified(
        df, num_clients=2, target_column='label', seed=42
    )

    # Same partitioning with same seed
    partitions2 = partition_data_stratified(
        df, num_clients=2, target_column='label', seed=42
    )

    # Check if results are identical
    identical = all(
        partitions1[i].equals(partitions2[i])
        for i in range(len(partitions1))
    )

    print(f"Same seed produces identical partitions: {identical}")

    # Different seed produces different partitions
    partitions3 = partition_data_stratified(
        df, num_clients=2, target_column='label', seed=99
    )

    different = not all(
        partitions1[i].equals(partitions3[i])
        for i in range(len(partitions1))
    )

    print(f"Different seed produces different partitions: {different}")


def example_error_handling():
    """Example: Error handling with validation."""
    print("\n=== Example 4: Error Handling ===")

    df = pd.DataFrame({
        'image_id': [1, 2, 3],
        'label': ['normal', 'pneumonia', 'normal']
    })

    # Test 1: Invalid num_clients
    try:
        partition_data_stratified(df, num_clients=0, target_column='label')
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Caught expected error (num_clients=0): {str(e)[:50]}...")

    # Test 2: Missing target_column
    try:
        partition_data_stratified(df, num_clients=2, target_column='nonexistent')
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Caught expected error (missing column): {str(e)[:50]}...")

    # Test 3: Invalid data type
    try:
        partition_data_stratified([1, 2, 3], num_clients=2, target_column='label')
        print("ERROR: Should have raised TypeError")
    except TypeError as e:
        print(f"Caught expected error (not DataFrame): {str(e)[:50]}...")


def example_integration_with_training():
    """Example: Integration with federated training workflow."""
    print("\n=== Example 5: Integration with Training Workflow ===")

    # Simulate loading metadata from CSV
    df = pd.DataFrame({
        'patient_id': list(range(1, 51)),
        'filename': [f'patient_{i}_xray.jpg' for i in range(1, 51)],
        'diagnosis': ['normal'] * 30 + ['pneumonia'] * 20,
    })

    print(f"Loaded {len(df)} patient records")

    # Distribute to federated clients
    num_federated_clients = 5
    client_datasets = partition_data_stratified(
        df,
        num_clients=num_federated_clients,
        target_column='diagnosis',
        seed=42
    )

    print(f"\nDataset prepared for {num_federated_clients} federated clients:")
    total_samples = 0
    for client_idx, client_data in enumerate(client_datasets):
        class_dist = client_data['diagnosis'].value_counts()
        print(f"  Client {client_idx}:")
        print(f"    Total samples: {len(client_data)}")
        print(f"    Normal: {class_dist.get('normal', 0)}, "
              f"Pneumonia: {class_dist.get('pneumonia', 0)}")
        total_samples += len(client_data)

    print(f"\nTotal samples distributed: {total_samples}")
    assert total_samples == len(df), "Data loss during partitioning!"
    print("Data integrity verified: All samples accounted for")


if __name__ == '__main__':
    example_basic_usage()
    example_with_logging()
    example_reproducibility()
    example_error_handling()
    example_integration_with_training()
    print("\n=== All examples completed successfully ===")
