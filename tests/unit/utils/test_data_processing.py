"""
Unit tests for DataProcessor utility class.
Tests data loading, sampling, splitting, and validation functionality.
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch
from federated_pneumonia_detection.src.internals.data_processing import DataProcessor
from federated_pneumonia_detection.models.system_constants import SystemConstants
from federated_pneumonia_detection.models.experiment_config import ExperimentConfig


class TestDataProcessor:
    """Test cases for DataProcessor utility."""

    @pytest.fixture
    def sample_constants(self):
        """Create sample SystemConstants for testing."""
        return SystemConstants.create_custom(
            base_path='test_data',
            metadata_filename='test_metadata.csv',
            main_images_folder='TestImages',
            images_subfolder='XRays'
        )

    @pytest.fixture
    def sample_config(self):
        """Create sample ExperimentConfig for testing."""
        return ExperimentConfig(
            sample_fraction=0.5,
            validation_split=0.2,
            seed=42
        )

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'patientId': ['001', '002', '003', '004', '005'],
            'Target': [0, 1, 0, 1, 1]
        })

    def test_init(self, sample_constants):
        """Test DataProcessor initialization."""
        processor = DataProcessor(sample_constants)
        assert processor.constants == sample_constants
        assert hasattr(processor, 'logger')

    @patch('federated_pneumonia_detection.src.utils.data_processing.pd.read_csv')
    @patch('federated_pneumonia_detection.src.utils.data_processing.Path.exists')
    def test_load_metadata_success(self, mock_exists, mock_read_csv, sample_constants, sample_dataframe):
        """Test successful metadata loading."""
        mock_exists.return_value = True
        mock_read_csv.return_value = sample_dataframe

        processor = DataProcessor(sample_constants)
        result = processor._load_metadata()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        # Path objects compare with pathlib, so we just check that read_csv was called
        mock_read_csv.assert_called_once()

    @patch('federated_pneumonia_detection.src.utils.data_processing.Path.exists')
    def test_load_metadata_file_not_found(self, mock_exists, sample_constants):
        """Test metadata loading when file doesn't exist."""
        mock_exists.return_value = False

        processor = DataProcessor(sample_constants)

        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            processor._load_metadata()

    @patch('federated_pneumonia_detection.src.utils.data_processing.pd.read_csv')
    @patch('federated_pneumonia_detection.src.utils.data_processing.Path.exists')
    def test_load_metadata_csv_error(self, mock_exists, mock_read_csv, sample_constants):
        """Test metadata loading when CSV reading fails."""
        mock_exists.return_value = True
        mock_read_csv.side_effect = pd.errors.EmptyDataError("Empty file")

        processor = DataProcessor(sample_constants)

        with pytest.raises(ValueError, match="Failed to load metadata file"):
            processor._load_metadata()

    def test_prepare_filenames(self, sample_constants, sample_dataframe):
        """Test filename preparation."""
        processor = DataProcessor(sample_constants)
        result = processor._prepare_filenames(sample_dataframe)

        expected_filenames = ['001.png', '002.png', '003.png', '004.png', '005.png']
        assert result['filename'].tolist() == expected_filenames
        assert result['Target'].dtype == 'object'  # Should be string

    def test_prepare_filenames_missing_patient_id(self, sample_constants):
        """Test filename preparation with missing patientId column."""
        df = pd.DataFrame({'wrongColumn': [1, 2, 3]})
        processor = DataProcessor(sample_constants)

        with pytest.raises(ValueError, match="Missing column: patientId"):
            processor._prepare_filenames(df)

    def test_validate_data_success(self, sample_constants):
        """Test successful data validation."""
        df = pd.DataFrame({
            'patientId': ['001', '002'],
            'Target': ['0', '1'],
            'filename': ['001.png', '002.png']
        })

        processor = DataProcessor(sample_constants)
        # Should not raise an exception
        processor._validate_data(df)

    def test_validate_data_missing_columns(self, sample_constants):
        """Test data validation with missing columns."""
        df = pd.DataFrame({'patientId': ['001', '002']})
        processor = DataProcessor(sample_constants)

        with pytest.raises(ValueError, match="Missing required columns"):
            processor._validate_data(df)

    def test_validate_data_empty_dataframe(self, sample_constants):
        """Test data validation with empty DataFrame."""
        df = pd.DataFrame()
        processor = DataProcessor(sample_constants)

        with pytest.raises(ValueError, match="DataFrame is empty"):
            processor._validate_data(df)

    def test_validate_data_missing_values(self, sample_constants):
        """Test data validation with missing values."""
        df = pd.DataFrame({
            'patientId': ['001', None],
            'Target': ['0', '1'],
            'filename': ['001.png', '002.png']
        })

        processor = DataProcessor(sample_constants)

        with pytest.raises(ValueError, match="Missing values found in column"):
            processor._validate_data(df)

    def test_sample_data_full_dataset(self, sample_constants, sample_dataframe):
        """Test data sampling when sample_fraction >= 1.0."""
        processor = DataProcessor(sample_constants)
        result = processor._sample_data(sample_dataframe, 1.0, 42)

        assert len(result) == len(sample_dataframe)
        assert result.equals(sample_dataframe)

    def test_sample_data_stratified(self, sample_constants, sample_dataframe):
        """Test stratified data sampling."""
        processor = DataProcessor(sample_constants)
        result = processor._sample_data(sample_dataframe, 0.6, 42)

        # Should maintain class distribution approximately
        assert len(result) < len(sample_dataframe)
        assert 'Target' in result.columns

    def test_sample_data_single_class(self, sample_constants):
        """Test data sampling with single class."""
        df = pd.DataFrame({
            'patientId': ['001', '002', '003', '004'],
            'Target': [0, 0, 0, 0]
        })

        processor = DataProcessor(sample_constants)
        result = processor._sample_data(df, 0.5, 42)

        assert len(result) < len(df)

    @patch('federated_pneumonia_detection.src.utils.data_processing.train_test_split')
    def test_create_train_val_split_stratified(self, mock_split, sample_constants, sample_dataframe):
        """Test train/validation split with stratification."""
        mock_split.return_value = (sample_dataframe.iloc[:3], sample_dataframe.iloc[3:])

        processor = DataProcessor(sample_constants)
        train_df, val_df = processor._create_train_val_split(sample_dataframe, 0.2, 42)

        assert len(train_df) == 3
        assert len(val_df) == 2
        mock_split.assert_called_once()

    def test_get_image_paths(self, sample_constants):
        """Test image path generation."""
        processor = DataProcessor(sample_constants)
        main_path, image_path = processor.get_image_paths()

        # Normalize paths for cross-platform comparison
        assert main_path == os.path.join('test_data', 'TestImages')
        assert image_path == os.path.join('test_data', 'TestImages', 'XRays')

    @patch('os.path.exists')
    def test_validate_image_paths_success(self, mock_exists, sample_constants):
        """Test successful image path validation."""
        mock_exists.return_value = True

        processor = DataProcessor(sample_constants)
        result = processor.validate_image_paths()

        assert result is True
        assert mock_exists.call_count == 2

    @patch('os.path.exists')
    def test_validate_image_paths_failure(self, mock_exists, sample_constants):
        """Test image path validation failure."""
        mock_exists.return_value = False

        processor = DataProcessor(sample_constants)
        result = processor.validate_image_paths()

        assert result is False

    @patch('federated_pneumonia_detection.src.utils.data_processing.load_and_split_data')
    def test_load_and_process_data_success(self, mock_load_and_split, sample_constants,
                                         sample_config, sample_dataframe):
        """Test complete data loading and processing pipeline."""
        # Setup mock to return train/val split
        mock_load_and_split.return_value = (sample_dataframe.iloc[:3], sample_dataframe.iloc[3:])

        processor = DataProcessor(sample_constants)
        train_df, val_df = processor.load_and_process_data(sample_config)

        assert len(train_df) == 3
        assert len(val_df) == 2

        # Verify the standalone function was called
        mock_load_and_split.assert_called_once_with(sample_constants, sample_config)

    @patch('federated_pneumonia_detection.src.utils.data_processing.load_and_split_data')
    def test_load_and_process_data_error_handling(self, mock_load_and_split, sample_constants, sample_config):
        """Test error handling in data processing pipeline."""
        mock_load_and_split.side_effect = FileNotFoundError("Test error")

        processor = DataProcessor(sample_constants)

        with pytest.raises(FileNotFoundError):
            processor.load_and_process_data(sample_config)