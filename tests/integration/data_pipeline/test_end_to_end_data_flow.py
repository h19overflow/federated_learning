"""
Integration tests for end-to-end data pipeline flow.
Tests complete data processing from raw files to training-ready datasets.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.src.internals.config_loader import ConfigLoader
from federated_pneumonia_detection.src.internals.data_processing import DataProcessor


class TestEndToEndDataFlow:
    """Integration tests for complete data pipeline."""

    @pytest.fixture
    def temp_data_structure(self):
        """Create temporary data structure for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create directory structure
            images_dir = temp_path / "Images" / "Images"
            images_dir.mkdir(parents=True)

            # Create sample images (empty files for testing)
            for i in range(10):
                image_file = images_dir / f"patient_{i:03d}.png"
                image_file.touch()

            # Create metadata CSV
            metadata_df = pd.DataFrame(
                {
                    "patientId": [f"patient_{i:03d}" for i in range(10)],
                    "Target": [i % 2 for i in range(10)],  # Alternating 0,1,0,1...
                    "age": [50 + i for i in range(10)],
                    "gender": ["M" if i % 2 == 0 else "F" for i in range(10)],
                },
            )

            metadata_path = temp_path / "Train_metadata.csv"
            metadata_df.to_csv(metadata_path, index=False)

            yield {
                "base_path": str(temp_path),
                "metadata_path": str(metadata_path),
                "images_dir": str(images_dir),
                "metadata_df": metadata_df,
            }

    def test_complete_data_pipeline(self, temp_data_structure):
        """Test complete data pipeline from config to train/val DataFrames."""
        # Setup constants with temp directory
        constants = SystemConstants.create_custom(
            base_path=temp_data_structure["base_path"],
            sample_fraction=0.8,  # Use 80% of data
            validation_split=0.25,  # 25% for validation
        )

        # Setup experiment config
        config = ExperimentConfig(sample_fraction=0.8, validation_split=0.25, seed=42)

        # Initialize data processor
        processor = DataProcessor(constants)

        # Test path validation
        assert processor.validate_image_paths() is True

        # Test complete processing pipeline
        train_df, val_df = processor.load_and_process_data(config)

        # Verify results
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert len(train_df) > 0
        assert len(val_df) > 0

        # Check that split ratio is approximately correct
        total_samples = len(train_df) + len(val_df)
        val_ratio = len(val_df) / total_samples
        assert 0.2 <= val_ratio <= 0.3  # Should be around 25%

        # Verify required columns exist
        required_columns = ["patientId", "Target", "filename"]
        for col in required_columns:
            assert col in train_df.columns
            assert col in val_df.columns

        # Verify filename format
        assert all(train_df["filename"].str.endswith(".png"))
        assert all(val_df["filename"].str.endswith(".png"))

        # Verify Target column is string type
        assert train_df["Target"].dtype == "object"
        assert val_df["Target"].dtype == "object"

    def test_config_integration(self, temp_data_structure):
        """Test integration between ConfigLoader and DataProcessor."""
        # Create temporary config file
        temp_config = {
            "system": {
                "sample_fraction": 0.6,
                "validation_split": 0.3,
                "batch_size": 64,
                "seed": 123,
            },
            "paths": {
                "base_path": temp_data_structure["base_path"],
                "metadata_filename": "Train_metadata.csv",
            },
            "columns": {
                "patient_id": "patientId",
                "target": "Target",
                "filename": "filename",
            },
        }

        # Test config loading and entity creation
        config_loader = ConfigLoader()
        constants = config_loader.create_system_constants(temp_config)
        exp_config = config_loader.create_experiment_config(temp_config)

        # Verify config values
        assert constants.SAMPLE_FRACTION == 0.6
        assert constants.BASE_PATH == temp_data_structure["base_path"]
        assert exp_config.validation_split == 0.3
        assert exp_config.seed == 123

        # Test processing with loaded config
        processor = DataProcessor(constants)
        train_df, val_df = processor.load_and_process_data(exp_config)

        assert len(train_df) > 0
        assert len(val_df) > 0

    def test_data_consistency_across_runs(self, temp_data_structure):
        """Test that data processing is reproducible with same seed."""
        constants = SystemConstants.create_custom(
            base_path=temp_data_structure["base_path"],
            seed=999,
            sample_fraction=1.0,  # Use all data to avoid empty samples
        )

        config = ExperimentConfig(seed=999, sample_fraction=1.0)
        processor = DataProcessor(constants)

        # Run processing twice with same seed
        train_df1, val_df1 = processor.load_and_process_data(config)
        train_df2, val_df2 = processor.load_and_process_data(config)

        # Results should be identical
        pd.testing.assert_frame_equal(train_df1, train_df2)
        pd.testing.assert_frame_equal(val_df1, val_df2)

    def test_class_balance_preservation(self, temp_data_structure):
        """Test that class balance is preserved during sampling and splitting."""
        constants = SystemConstants.create_custom(
            base_path=temp_data_structure["base_path"],
            sample_fraction=1.0,  # Use all data
        )

        config = ExperimentConfig(sample_fraction=1.0, seed=42)
        processor = DataProcessor(constants)

        train_df, val_df = processor.load_and_process_data(config)

        # Check class distribution
        train_class_counts = train_df["Target"].value_counts()
        val_class_counts = val_df["Target"].value_counts()

        # Both classes should be present in both splits
        assert len(train_class_counts) == 2
        assert len(val_class_counts) == 2

        # Class balance should be roughly maintained
        train_balance = train_class_counts.min() / train_class_counts.max()
        val_balance = val_class_counts.min() / val_class_counts.max()

        # Allow some imbalance but not extreme
        assert train_balance >= 0.3
        assert val_balance >= 0.3

    def test_error_handling_missing_files(self, temp_data_structure):
        """Test error handling when expected files are missing."""
        # Remove metadata file
        os.remove(temp_data_structure["metadata_path"])

        constants = SystemConstants.create_custom(
            base_path=temp_data_structure["base_path"],
        )
        config = ExperimentConfig()
        processor = DataProcessor(constants)

        # Should raise an error when metadata file is missing
        with pytest.raises((FileNotFoundError, ValueError)):
            processor.load_and_process_data(config)

    def test_edge_case_small_dataset(self, temp_data_structure):
        """Test handling of very small datasets."""
        # Create minimal dataset with 4 samples (2 per class) - minimum for stratified split
        minimal_metadata = pd.DataFrame(
            {
                "patientId": [
                    "patient_001",
                    "patient_002",
                    "patient_003",
                    "patient_004",
                ],
                "Target": [0, 0, 1, 1],
            },
        )

        metadata_path = Path(temp_data_structure["base_path"]) / "Train_metadata.csv"
        minimal_metadata.to_csv(metadata_path, index=False)

        constants = SystemConstants.create_custom(
            base_path=temp_data_structure["base_path"],
            validation_split=0.5,  # 50% split for 4 samples = 2 each
        )

        config = ExperimentConfig(validation_split=0.5, sample_fraction=1.0)
        processor = DataProcessor(constants)

        train_df, val_df = processor.load_and_process_data(config)

        # Should still work with minimal data
        assert len(train_df) >= 1
        assert len(val_df) >= 1
        assert len(train_df) + len(val_df) == 4
