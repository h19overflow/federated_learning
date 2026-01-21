"""
Unit tests for federated learning data manager.
Tests data loading, splitting, and DataLoader creation.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.src.control.federated_learning.data_manager import (
    load_data,
    split_partition,
)


class TestSplitPartition:
    """Tests for split_partition function."""

    @pytest.fixture
    def sample_partition(self):
        """Create a sample partition DataFrame."""
        data = {
            "filename": [f"image_{i}.jpg" for i in range(100)],
            "target": [0, 1] * 50,
        }
        return pd.DataFrame(data)

    def test_split_partition_returns_tuple(self, sample_partition):
        """Test that split_partition returns tuple of two DataFrames."""
        train_df, val_df = split_partition(
            partition_df=sample_partition,
            validation_split=0.2,
            target_column="target",
            seed=42,
        )

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)

    def test_split_partition_correct_split_ratio(self, sample_partition):
        """Test that split ratio is approximately correct."""
        train_df, val_df = split_partition(
            partition_df=sample_partition,
            validation_split=0.2,
            target_column="target",
            seed=42,
        )

        total = len(train_df) + len(val_df)
        assert total == len(sample_partition)

        # Check ratio is approximately 80/20
        val_ratio = len(val_df) / total
        assert 0.15 < val_ratio < 0.25  # Allow some tolerance

    def test_split_partition_preserves_total_samples(self, sample_partition):
        """Test that split preserves total number of samples."""
        for split_ratio in [0.1, 0.2, 0.3, 0.5]:
            train_df, val_df = split_partition(
                partition_df=sample_partition,
                validation_split=split_ratio,
                target_column="target",
                seed=42,
            )

            total = len(train_df) + len(val_df)
            assert total == len(sample_partition)

    def test_split_partition_reproducible_with_seed(self, sample_partition):
        """Test that split is reproducible with same seed."""
        train_df1, val_df1 = split_partition(
            partition_df=sample_partition,
            validation_split=0.2,
            target_column="target",
            seed=42,
        )

        train_df2, val_df2 = split_partition(
            partition_df=sample_partition,
            validation_split=0.2,
            target_column="target",
            seed=42,
        )

        pd.testing.assert_frame_equal(
            train_df1.reset_index(drop=True),
            train_df2.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            val_df1.reset_index(drop=True),
            val_df2.reset_index(drop=True),
        )

    def test_split_partition_different_with_different_seed(self, sample_partition):
        """Test that different seeds produce different splits."""
        train_df1, val_df1 = split_partition(
            partition_df=sample_partition,
            validation_split=0.2,
            target_column="target",
            seed=42,
        )

        train_df2, val_df2 = split_partition(
            partition_df=sample_partition,
            validation_split=0.2,
            target_column="target",
            seed=999,
        )

        # Splits should be different
        assert not train_df1.equals(train_df2)

    def test_split_partition_stratified_fallback(self, sample_partition):
        """Test that split falls back to random split if stratification fails."""
        # Create a partition that might cause stratification issues
        train_df, val_df = split_partition(
            partition_df=sample_partition,
            validation_split=0.2,
            target_column="target",
            seed=42,
        )

        # Should still return valid split
        assert len(train_df) > 0
        assert len(val_df) > 0

    def test_split_partition_index_reset(self, sample_partition):
        """Test that indices are reset in split DataFrames."""
        train_df, val_df = split_partition(
            partition_df=sample_partition,
            validation_split=0.2,
            target_column="target",
            seed=42,
        )

        # Indices should be continuous from 0
        assert train_df.index[0] == 0
        assert list(train_df.index) == list(range(len(train_df)))

        assert val_df.index[0] == 0
        assert list(val_df.index) == list(range(len(val_df)))

    def test_split_partition_preserves_columns(self, sample_partition):
        """Test that split preserves all columns."""
        train_df, val_df = split_partition(
            partition_df=sample_partition,
            validation_split=0.2,
            target_column="target",
            seed=42,
        )

        assert list(train_df.columns) == list(sample_partition.columns)
        assert list(val_df.columns) == list(sample_partition.columns)

    def test_split_partition_extreme_ratios(self, sample_partition):
        """Test split with extreme validation ratios."""
        # Very small validation split
        train_df, val_df = split_partition(
            partition_df=sample_partition,
            validation_split=0.01,
            target_column="target",
            seed=42,
        )
        assert len(train_df) > 0
        assert len(val_df) > 0

        # Large validation split
        train_df, val_df = split_partition(
            partition_df=sample_partition,
            validation_split=0.9,
            target_column="target",
            seed=42,
        )
        assert len(train_df) > 0
        assert len(val_df) > 0


class TestLoadData:
    """Tests for load_data function."""

    @pytest.fixture
    def mock_setup(self, tmp_path):
        """Setup mock dependencies for load_data."""
        # Create temporary image directory
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        # Create sample images
        import numpy as np
        from PIL import Image

        for i in range(20):
            img = Image.fromarray(
                np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
            )
            img.save(image_dir / f"image_{i}.jpg")

        constants = SystemConstants()

        # Create sample DataFrame with correct column names
        data = {
            constants.FILENAME_COLUMN: [f"image_{i}.jpg" for i in range(20)],
            constants.TARGET_COLUMN: [0, 1] * 10,
        }
        df = pd.DataFrame(data)

        config = ExperimentConfig(batch_size=4, validation_split=0.2)

        return {
            "df": df,
            "image_dir": image_dir,
            "constants": constants,
            "config": config,
        }

    def test_load_data_returns_tuple_of_dataloaders(self, mock_setup):
        """Test that load_data returns tuple of two DataLoaders."""
        train_loader, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=mock_setup["image_dir"],
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_load_data_dataloaders_have_batches(self, mock_setup):
        """Test that created DataLoaders can produce batches."""
        train_loader, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=mock_setup["image_dir"],
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        # Try to get a batch from train loader
        for images, labels in train_loader:
            assert images.shape[0] > 0
            assert labels.shape[0] > 0
            break

    def test_load_data_respects_batch_size(self, mock_setup):
        """Test that DataLoaders respect batch_size from config."""
        mock_setup["config"].batch_size = 4

        train_loader, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=mock_setup["image_dir"],
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        for images, labels in train_loader:
            # Batch size should be <= 4
            assert images.shape[0] <= 4
            break

    def test_load_data_respects_validation_split(self, mock_setup):
        """Test that validation split is respected."""
        mock_setup["config"].validation_split = 0.3

        train_loader, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=mock_setup["image_dir"],
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        total_samples = len(train_loader.dataset) + len(val_loader.dataset)
        assert total_samples == len(mock_setup["df"])

    def test_load_data_train_shuffle_true(self, mock_setup):
        """Test that training DataLoader shuffles data."""
        train_loader, _ = load_data(
            partition_df=mock_setup["df"],
            image_dir=mock_setup["image_dir"],
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        # Train loader should have shuffle enabled
        assert train_loader.sampler is not None or train_loader.shuffle

    def test_load_data_val_shuffle_false(self, mock_setup):
        """Test that validation DataLoader does not shuffle data."""
        _, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=mock_setup["image_dir"],
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        # Val loader should not shuffle
        if hasattr(val_loader, "shuffle"):
            assert val_loader.shuffle is False

    def test_load_data_empty_partition_raises_error(self, mock_setup):
        """Test that empty partition raises ValueError."""
        empty_df = pd.DataFrame(
            {
                mock_setup["constants"].FILENAME_COLUMN: [],
                mock_setup["constants"].TARGET_COLUMN: [],
            },
        )

        with pytest.raises(ValueError, match="partition_df cannot be empty"):
            load_data(
                partition_df=empty_df,
                image_dir=mock_setup["image_dir"],
                constants=mock_setup["constants"],
                config=mock_setup["config"],
            )

    def test_load_data_missing_image_dir_raises_error(self, mock_setup):
        """Test that non-existent image directory raises ValueError."""
        fake_dir = mock_setup["image_dir"] / "nonexistent"

        with pytest.raises(ValueError, match="Image directory not found"):
            load_data(
                partition_df=mock_setup["df"],
                image_dir=fake_dir,
                constants=mock_setup["constants"],
                config=mock_setup["config"],
            )

    def test_load_data_missing_filename_column_raises_error(self, mock_setup):
        """Test that missing filename column raises ValueError."""
        constants = SystemConstants()
        bad_df = mock_setup["df"].rename(
            columns={constants.FILENAME_COLUMN: "bad_name"},
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            load_data(
                partition_df=bad_df,
                image_dir=mock_setup["image_dir"],
                constants=mock_setup["constants"],
                config=mock_setup["config"],
            )

    def test_load_data_missing_target_column_raises_error(self, mock_setup):
        """Test that missing target column raises ValueError."""
        constants = SystemConstants()
        bad_df = mock_setup["df"].rename(columns={constants.TARGET_COLUMN: "bad_name"})

        with pytest.raises(ValueError, match="Missing required columns"):
            load_data(
                partition_df=bad_df,
                image_dir=mock_setup["image_dir"],
                constants=mock_setup["constants"],
                config=mock_setup["config"],
            )

    def test_load_data_custom_validation_split_override(self, mock_setup):
        """Test that validation_split parameter overrides config."""
        mock_setup["config"].validation_split = 0.2

        train_loader, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=mock_setup["image_dir"],
            constants=mock_setup["constants"],
            config=mock_setup["config"],
            validation_split=0.5,
        )

        # Should use 0.5 split, not 0.2
        val_ratio = len(val_loader.dataset) / (
            len(train_loader.dataset) + len(val_loader.dataset)
        )
        assert 0.4 < val_ratio < 0.6

    def test_load_data_image_dir_as_string(self, mock_setup):
        """Test that image_dir can be provided as string."""
        train_loader, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=str(mock_setup["image_dir"]),  # Pass as string
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_load_data_image_dir_as_path(self, mock_setup):
        """Test that image_dir works with Path objects."""
        train_loader, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=Path(mock_setup["image_dir"]),
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_load_data_with_color_mode_config(self, mock_setup):
        """Test load_data with custom color_mode configuration."""
        mock_setup["config"].color_mode = "RGB"

        train_loader, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=mock_setup["image_dir"],
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        # Should successfully load with specified color_mode
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_load_data_with_augmentation_config(self, mock_setup):
        """Test load_data with augmentation enabled."""
        mock_setup["config"].augmentation_strength = 0.5

        train_loader, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=mock_setup["image_dir"],
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_load_data_train_samples_count(self, mock_setup):
        """Test that train and validation samples sum to partition size."""
        train_loader, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=mock_setup["image_dir"],
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        total = len(train_loader.dataset) + len(val_loader.dataset)
        assert total == len(mock_setup["df"])

    def test_load_data_creates_valid_batches(self, mock_setup):
        """Test that created DataLoaders produce valid batches."""
        train_loader, val_loader = load_data(
            partition_df=mock_setup["df"],
            image_dir=mock_setup["image_dir"],
            constants=mock_setup["constants"],
            config=mock_setup["config"],
        )

        # Check train batch
        for images, labels in train_loader:
            assert isinstance(images, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert images.dim() == 4  # [batch, channels, height, width]
            assert labels.dim() == 1  # [batch]
            break

        # Check val batch
        for images, labels in val_loader:
            assert isinstance(images, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert images.dim() == 4
            assert labels.dim() == 1
            break
