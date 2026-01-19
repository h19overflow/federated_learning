"""
Shared fixtures and configuration for pytest.
Provides common test setup and utilities across all test modules.
"""

import pytest
import pandas as pd
import tempfile
import os
import logging
from typing import Dict, Generator

from federated_pneumonia_detection.config.config_manager import ConfigManager
from federated_pneumonia_detection.src.internals.data_processing import DataProcessor

from tests.fixtures.sample_data import (
    SampleDataFactory,
    TempDataStructure,
    MockDatasets,
    create_test_config_dict
)


# Configure test logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests


@pytest.fixture(scope="session")
def test_config_dict() -> Dict:
    """Session-wide test configuration dictionary."""
    return create_test_config_dict()


@pytest.fixture
def sample_config() -> ConfigManager:
    """Create sample ConfigManager for testing."""
    return ConfigManager()


@pytest.fixture
def sample_constants(sample_config) -> ConfigManager:
    """
    Create sample constants for testing.
    DEPRECATED: Use sample_config instead.
    """
    return sample_config


@pytest.fixture
def sample_experiment_config(sample_config) -> ConfigManager:
    """
    Create sample ExperimentConfig for testing.
    DEPRECATED: Use sample_config instead.
    """
    return sample_config


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    return SampleDataFactory.create_sample_metadata(num_samples=50)


@pytest.fixture
def minimal_dataframe() -> pd.DataFrame:
    """Create minimal DataFrame for edge case testing."""
    return SampleDataFactory.create_minimal_metadata()


@pytest.fixture
def imbalanced_dataframe() -> pd.DataFrame:
    """Create imbalanced DataFrame for testing."""
    return SampleDataFactory.create_imbalanced_metadata(num_samples=100)


@pytest.fixture
def single_class_dataframe() -> pd.DataFrame:
    """Create single-class DataFrame for testing."""
    return SampleDataFactory.create_single_class_metadata(num_samples=30)


@pytest.fixture
def corrupted_dataframes() -> Dict[str, pd.DataFrame]:
    """Create various corrupted DataFrames for error testing."""
    return SampleDataFactory.create_corrupted_metadata()


@pytest.fixture
def temp_data_structure() -> Generator[Dict[str, str], None, None]:
    """Create temporary data structure with files."""
    with TempDataStructure() as paths:
        yield paths


@pytest.fixture
def temp_data_structure_custom(request) -> Generator[Dict[str, str], None, None]:
    """Create temporary data structure with custom parameters."""
    # Get parameters from test request
    metadata_df = getattr(request, 'param', {}).get('metadata_df')
    create_images = getattr(request, 'param', {}).get('create_images', True)

    with TempDataStructure(
        metadata_df=metadata_df,
        create_images=create_images
    ) as paths:
        yield paths


@pytest.fixture
def pneumonia_dataset() -> pd.DataFrame:
    """Create realistic pneumonia dataset."""
    return MockDatasets.pneumonia_dataset()


@pytest.fixture
def federated_datasets() -> list:
    """Create federated learning datasets."""
    return MockDatasets.federated_datasets()


@pytest.fixture
def data_processor(sample_config) -> DataProcessor:
    """Create DataProcessor instance with sample config."""
    return DataProcessor(sample_config)


@pytest.fixture
def temp_config_file() -> Generator[str, None, None]:
    """Create temporary configuration file."""
    config_content = """
# Test configuration
system:
  img_size: [224, 224]
  batch_size: 16
  sample_fraction: 0.3
  seed: 123

experiment:
  learning_rate: 0.01
  epochs: 2

paths:
  base_path: "test_data"
  metadata_filename: "test_metadata.csv"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_file = f.name

    yield temp_file

    # Cleanup
    os.unlink(temp_file)


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Automatically setup test environment for all tests."""
    # Set environment variables for testing
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    # Disable GPU usage during testing
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="function")
def isolated_temp_dir():
    """Create isolated temporary directory for each test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_logger():
    """Create mock logger for testing."""
    import logging
    from unittest.mock import Mock

    mock_logger = Mock(spec=logging.Logger)
    mock_logger.info = Mock()
    mock_logger.warning = Mock()
    mock_logger.error = Mock()
    mock_logger.debug = Mock()

    return mock_logger


# Helper functions for tests
def assert_dataframe_structure(df: pd.DataFrame, required_columns: list, min_rows: int = 1):
    """Assert that DataFrame has required structure."""
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= min_rows
    for col in required_columns:
        assert col in df.columns
    assert not df.empty


def assert_valid_split(train_df: pd.DataFrame, val_df: pd.DataFrame,
                      expected_split_ratio: float = 0.2, tolerance: float = 0.1):
    """Assert that train/validation split is reasonable."""
    total_samples = len(train_df) + len(val_df)
    actual_val_ratio = len(val_df) / total_samples

    assert abs(actual_val_ratio - expected_split_ratio) <= tolerance
    assert len(train_df) > 0
    assert len(val_df) > 0


# Parametrized fixtures for testing multiple scenarios
@pytest.fixture(params=[
    {'sample_fraction': 0.1, 'validation_split': 0.2},
    {'sample_fraction': 0.5, 'validation_split': 0.3},
    {'sample_fraction': 1.0, 'validation_split': 0.25}
])
def data_processing_params(request):
    """Parametrized data processing parameters."""
    return request.param


@pytest.fixture(params=[42, 123, 999])
def test_seeds(request):
    """Different random seeds for reproducibility testing."""
    return request.param


@pytest.fixture(params=[
    (224, 224),
    (256, 256),
    (512, 512)
])
def image_sizes(request):
    """Different image sizes for testing."""
    return request.param