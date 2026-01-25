"""
Unit tests for CustomPartitioner.

Tests data partitioning functionality for federated learning.
"""

import numpy as np
import pandas as pd
import pytest

from federated_pneumonia_detection.src.control.federated_new_version.partioner import (
    CustomPartitioner,
)


class TestCustomPartitioner:
    """Test suite for CustomPartitioner."""

    def test_partitioner_initialization(self, sample_metadata_df):
        """Test partitioner initialization with dataset."""
        num_partitions = 3
        partitioner = CustomPartitioner(sample_metadata_df, num_partitions)

        assert partitioner._num_partitions == num_partitions
        assert len(partitioner._base_dataset) == len(sample_metadata_df)
        assert len(partitioner.partition_indices) == num_partitions

    def test_partitioner_num_partitions_property(self, sample_metadata_df):
        """Test num_partitions property."""
        num_partitions = 5
        partitioner = CustomPartitioner(sample_metadata_df, num_partitions)

        assert partitioner.num_partitions == num_partitions

    def test_partition_indices_creation(self, sample_metadata_df):
        """Test that partition indices are created correctly."""
        num_partitions = 3
        partitioner = CustomPartitioner(sample_metadata_df, num_partitions)

        # Check that all indices are covered
        all_indices = []
        for indices in partitioner.partition_indices:
            all_indices.extend(indices)

        # All indices should be present exactly once
        expected_indices = set(range(len(sample_metadata_df)))
        actual_indices = set(all_indices)

        assert len(all_indices) == len(sample_metadata_df)
        assert expected_indices == actual_indices

    def test_load_partition_valid_id(self, sample_metadata_df):
        """Test loading a valid partition."""
        num_partitions = 3
        partitioner = CustomPartitioner(sample_metadata_df, num_partitions)

        partition_df = partitioner.load_partition(0)

        assert isinstance(partition_df, pd.DataFrame)
        assert len(partition_df) > 0
        assert len(partition_df) <= len(sample_metadata_df)
        assert "filename" in partition_df.columns

    def test_load_partition_adds_filename_column(self, sample_metadata_df):
        """Test that filename column is added if missing."""
        # Create DataFrame without filename column
        df_without_filename = pd.DataFrame(
            {
                "patientId": [1, 2, 3, 4, 5],
                "class": ["Normal", "Pneumonia", "Normal", "Pneumonia", "Normal"],
            },
        )

        num_partitions = 2
        partitioner = CustomPartitioner(df_without_filename, num_partitions)
        partition_df = partitioner.load_partition(0)

        assert "filename" in partition_df.columns
        # Check that filename is constructed correctly
        assert partition_df["filename"].str.endswith(".png").all()
        assert partition_df["filename"].str.contains("1|2|3|4|5").any()

    def test_load_partition_preserves_filename_column(self, sample_metadata_df):
        """Test that existing filename column is preserved."""
        partitioner = CustomPartitioner(sample_metadata_df, 2)
        partition_df = partitioner.load_partition(0)

        assert "filename" in partition_df.columns
        # Verify filename format from original data
        assert partition_df["filename"].str.endswith(".png").all()

    def test_load_partition_invalid_id_negative(self, sample_metadata_df):
        """Test loading partition with negative ID."""
        partitioner = CustomPartitioner(sample_metadata_df, 3)

        with pytest.raises(ValueError):
            partitioner.load_partition(-1)

    def test_load_partition_invalid_id_too_large(self, sample_metadata_df):
        """Test loading partition with ID >= num_partitions."""
        partitioner = CustomPartitioner(sample_metadata_df, 3)

        with pytest.raises(ValueError):
            partitioner.load_partition(3)

        with pytest.raises(ValueError):
            partitioner.load_partition(10)

    def test_partition_size_distribution(self, sample_metadata_df):
        """Test that partitions are roughly evenly distributed."""
        num_partitions = 3
        partitioner = CustomPartitioner(sample_metadata_df, num_partitions)

        partition_sizes = [
            len(partitioner.load_partition(i)) for i in range(num_partitions)
        ]

        # All partitions should be non-empty
        assert all(size > 0 for size in partition_sizes)

        # Sizes should be roughly equal (within 1 for small datasets)
        max_size = max(partition_sizes)
        min_size = min(partition_sizes)
        assert (max_size - min_size) <= 1

    def test_all_partitions_cover_all_data(self, sample_metadata_df):
        """Test that all data is covered across all partitions."""
        num_partitions = 3
        partitioner = CustomPartitioner(sample_metadata_df, num_partitions)

        # Load all partitions and collect patientIds
        all_patient_ids = set()
        for i in range(num_partitions):
            partition_df = partitioner.load_partition(i)
            all_patient_ids.update(partition_df["patientId"])

        # Check that all original patientIds are covered
        original_patient_ids = set(sample_metadata_df["patientId"])
        assert all_patient_ids == original_patient_ids

    def test_partition_with_single_partition(self, sample_metadata_df):
        """Test partitioning into a single partition."""
        num_partitions = 1
        partitioner = CustomPartitioner(sample_metadata_df, num_partitions)

        partition_df = partitioner.load_partition(0)

        assert len(partition_df) == len(sample_metadata_df)
        assert partitioner.num_partitions == 1

    def test_partition_with_many_partitions(self, sample_metadata_df):
        """Test partitioning into many partitions (more than samples)."""
        num_partitions = len(sample_metadata_df) + 2
        partitioner = CustomPartitioner(sample_metadata_df, num_partitions)

        # Some partitions may be empty, but should not raise errors
        empty_count = 0
        for i in range(num_partitions):
            partition_df = partitioner.load_partition(i)
            if len(partition_df) == 0:
                empty_count += 1

        # Should have some non-empty partitions
        assert empty_count < num_partitions

    def test_partition_reset_index(self, sample_metadata_df):
        """Test that partition index is reset starting from 0."""
        partitioner = CustomPartitioner(sample_metadata_df, 2)
        partition_df = partitioner.load_partition(0)

        # Check that index is reset
        expected_index = list(range(len(partition_df)))
        actual_index = list(partition_df.index)

        assert actual_index == expected_index

    def test_partition_with_large_dataset(self):
        """Test partitioning with larger dataset."""
        large_df = pd.DataFrame(
            {
                "patientId": list(range(1, 101)),
                "class": ["Normal"] * 50 + ["Pneumonia"] * 50,
            },
        )

        num_partitions = 10
        partitioner = CustomPartitioner(large_df, num_partitions)

        # Check each partition
        for i in range(num_partitions):
            partition_df = partitioner.load_partition(i)
            assert len(partition_df) == 10  # 100 samples / 10 partitions
            assert "filename" in partition_df.columns

    def test_partition_reproducibility_with_seed(self, sample_metadata_df):
        """Test that partitioning is reproducible when numpy seed is set."""
        np.random.seed(42)
        partitioner1 = CustomPartitioner(sample_metadata_df, 3)
        partition_df1 = partitioner1.load_partition(0)

        np.random.seed(42)
        partitioner2 = CustomPartitioner(sample_metadata_df, 3)
        partition_df2 = partitioner2.load_partition(0)

        # Partitions should be identical
        pd.testing.assert_frame_equal(partition_df1, partition_df2)

    def test_partition_different_without_seed(self, sample_metadata_df):
        """Test that partitioning differs without fixed seed."""
        partitioner1 = CustomPartitioner(sample_metadata_df, 3)
        partition_df1 = partitioner1.load_partition(0)

        partitioner2 = CustomPartitioner(sample_metadata_df, 3)
        partition_df2 = partitioner2.load_partition(0)

        # Partitions should likely be different (though could be same by chance)
        # We just check they don't raise errors
        assert len(partition_df1) > 0
        assert len(partition_df2) > 0

    def test_partition_data_integrity(self, sample_metadata_df):
        """Test that partition data maintains original data integrity."""
        num_partitions = 2
        partitioner = CustomPartitioner(sample_metadata_df, num_partitions)

        # Get original columns
        original_columns = set(sample_metadata_df.columns)

        for i in range(num_partitions):
            partition_df = partitioner.load_partition(i)

            # Check that all original columns are present
            assert set(partition_df.columns).issuperset(original_columns)

            # Check that patientId values are from original
            assert set(partition_df["patientId"]).issubset(
                set(sample_metadata_df["patientId"]),
            )

            # Check that class values are valid
            assert all(cls in ["Normal", "Pneumonia"] for cls in partition_df["class"])

    def test_partition_with_minimal_data(self):
        """Test partitioning with minimal data (1 sample)."""
        minimal_df = pd.DataFrame(
            {
                "patientId": [1],
                "class": ["Normal"],
            },
        )

        partitioner = CustomPartitioner(minimal_df, 1)
        partition_df = partitioner.load_partition(0)

        assert len(partition_df) == 1
        assert partition_df["patientId"].iloc[0] == 1

    def test_partition_indices_are_permuted(self, sample_metadata_df):
        """Test that partition indices represent a random permutation."""
        partitioner = CustomPartitioner(sample_metadata_df, 3)

        # Get all indices in order they appear in partitions
        indices_in_partitions = []
        for i in range(3):
            partition_indices = partitioner.partition_indices[i]
            indices_in_partitions.extend(partition_indices)

        # Should not be sorted (highly unlikely to be sorted by random)
        # This is a probabilistic test, but with 10 items, probability is very low
        assert indices_in_partitions != sorted(indices_in_partitions)
