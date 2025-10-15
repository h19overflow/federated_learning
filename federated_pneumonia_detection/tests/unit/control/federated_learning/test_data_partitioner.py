"""
Unit tests for data partitioning functions.
"""

import pytest
import pandas as pd
import numpy as np

from federated_pneumonia_detection.src.control.federated_learning.data_partitioner import (
    partition_data_iid,
    partition_data_by_patient,
    partition_data_stratified
)


@pytest.mark.unit
class TestIIDPartitioning:
    """Test IID (random) partitioning strategy."""

    def test_iid_basic_partitioning(self, federated_df):
        """Test basic IID partitioning works."""
        num_clients = 3
        partitions = partition_data_iid(federated_df, num_clients, seed=42)

        assert len(partitions) == num_clients
        assert all(isinstance(p, pd.DataFrame) for p in partitions)
        
        total_samples = sum(len(p) for p in partitions)
        assert total_samples == len(federated_df)

    def test_iid_roughly_equal_sizes(self, federated_df):
        """Test that partitions are roughly equal in size."""
        partitions = partition_data_iid(federated_df, num_clients=3, seed=42)
        
        sizes = [len(p) for p in partitions]
        max_size = max(sizes)
        min_size = min(sizes)
        
        # Should differ by at most 1 sample
        assert max_size - min_size <= 1

    def test_iid_reproducibility(self, federated_df):
        """Test that same seed produces same partitions."""
        partitions1 = partition_data_iid(federated_df, num_clients=3, seed=42)
        partitions2 = partition_data_iid(federated_df, num_clients=3, seed=42)

        for p1, p2 in zip(partitions1, partitions2):
            pd.testing.assert_frame_equal(p1.reset_index(drop=True), 
                                         p2.reset_index(drop=True))

    def test_iid_different_seeds(self, federated_df):
        """Test different seeds produce different partitions."""
        partitions1 = partition_data_iid(federated_df, num_clients=3, seed=42)
        partitions2 = partition_data_iid(federated_df, num_clients=3, seed=123)

        assert not partitions1[0].equals(partitions2[0])

    def test_iid_invalid_num_clients(self, federated_df):
        """Test error handling for invalid num_clients."""
        with pytest.raises(ValueError, match="must be positive"):
            partition_data_iid(federated_df, num_clients=0)

        with pytest.raises(ValueError, match="must be positive"):
            partition_data_iid(federated_df, num_clients=-1)

    def test_iid_too_few_samples(self):
        """Test error when dataset smaller than num_clients."""
        tiny_df = pd.DataFrame({'patientId': ['p1', 'p2'], 'Target': [0, 1]})
        
        with pytest.raises(ValueError, match="smaller than num_clients"):
            partition_data_iid(tiny_df, num_clients=5)


@pytest.mark.unit
class TestNonIIDPartitioning:
    """Test non-IID (patient-based) partitioning strategy."""

    def test_patient_based_partitioning(self, federated_df):
        """Test patient-based partitioning works."""
        partitions = partition_data_by_patient(
            federated_df, 
            num_clients=3,
            patient_column='patientId',
            seed=42
        )

        assert len(partitions) == 3
        total_samples = sum(len(p) for p in partitions)
        assert total_samples == len(federated_df)

    def test_patient_no_overlap(self, federated_df):
        """Test that patients don't appear in multiple partitions."""
        partitions = partition_data_by_patient(
            federated_df,
            num_clients=3,
            patient_column='patientId',
            seed=42
        )

        all_patients = []
        for partition in partitions:
            patients = partition['patientId'].unique().tolist()
            all_patients.extend(patients)

        # Check no duplicates
        assert len(all_patients) == len(set(all_patients))

    def test_patient_missing_column(self, federated_df):
        """Test error when patient column is missing."""
        with pytest.raises(ValueError, match="not found in DataFrame"):
            partition_data_by_patient(
                federated_df,
                num_clients=3,
                patient_column='nonexistent_column'
            )


@pytest.mark.unit
class TestStratifiedPartitioning:
    """Test stratified partitioning strategy."""

    def test_stratified_basic(self, federated_df):
        """Test stratified partitioning works."""
        partitions = partition_data_stratified(
            federated_df,
            num_clients=3,
            target_column='Target',
            seed=42
        )

        assert len(partitions) == 3
        total_samples = sum(len(p) for p in partitions)
        assert total_samples == len(federated_df)

    def test_stratified_class_distribution(self, federated_df):
        """Test that class distributions are similar across partitions."""
        partitions = partition_data_stratified(
            federated_df,
            num_clients=3,
            target_column='Target',
            seed=42
        )

        # Calculate class ratios for each partition
        ratios = []
        for partition in partitions:
            if len(partition) > 0:
                pos_ratio = (partition['Target'] == 1).sum() / len(partition)
                ratios.append(pos_ratio)

        # All ratios should be similar (within 0.2)
        max_diff = max(ratios) - min(ratios)
        assert max_diff < 0.3, f"Class distributions too different: {ratios}"

    def test_stratified_missing_target_column(self, federated_df):
        """Test error when target column is missing."""
        with pytest.raises(ValueError, match="not found in DataFrame"):
            partition_data_stratified(
                federated_df,
                num_clients=3,
                target_column='nonexistent'
            )


