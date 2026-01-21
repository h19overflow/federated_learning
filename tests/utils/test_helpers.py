"""
Test helper utilities and common testing functions.
Provides reusable utilities for testing across all test modules.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union
from unittest.mock import Mock

import numpy as np
import pandas as pd

from federated_pneumonia_detection.models.experiment_config import ExperimentConfig
from federated_pneumonia_detection.models.system_constants import SystemConstants


class TestHelpers:
    """Collection of helper methods for testing."""

    @staticmethod
    def assert_dataframe_valid(
        df: pd.DataFrame,
        required_columns: List[str],
        min_rows: int = 1,
        check_nulls: bool = True,
    ) -> None:
        """
        Assert that DataFrame meets basic validity requirements.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows expected
            check_nulls: Whether to check for null values
        """
        assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
        assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"
        assert not df.empty, "DataFrame should not be empty"

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

        if check_nulls:
            for col in required_columns:
                assert not df[col].isna().any(), f"Found null values in column: {col}"

    @staticmethod
    def assert_train_val_split_valid(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        expected_split: float = 0.2,
        tolerance: float = 0.1,
    ) -> None:
        """
        Assert that train/validation split is reasonable.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            expected_split: Expected validation split ratio
            tolerance: Acceptable deviation from expected split
        """
        total_samples = len(train_df) + len(val_df)
        actual_split = len(val_df) / total_samples

        assert len(train_df) > 0, "Training set should not be empty"
        assert len(val_df) > 0, "Validation set should not be empty"
        assert abs(actual_split - expected_split) <= tolerance, (
            f"Split ratio {actual_split:.3f} deviates too much from expected {expected_split:.3f}"
        )

    @staticmethod
    def create_mock_logger() -> Mock:
        """Create a mock logger for testing."""
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.warning = Mock()
        mock_logger.error = Mock()
        mock_logger.debug = Mock()
        return mock_logger

    @staticmethod
    def create_temp_image_files(
        directory: Union[str, Path],
        patient_ids: List[str],
        extension: str = ".png",
    ) -> List[Path]:
        """
        Create temporary empty image files for testing.

        Args:
            directory: Directory to create files in
            patient_ids: List of patient IDs for filenames
            extension: File extension

        Returns:
            List of created file paths
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        created_files = []
        for patient_id in patient_ids:
            file_path = directory / f"{patient_id}{extension}"
            file_path.touch()
            created_files.append(file_path)

        return created_files

    @staticmethod
    def create_test_metadata_csv(
        file_path: Union[str, Path],
        num_samples: int = 20,
        class_balance: float = 0.5,
    ) -> pd.DataFrame:
        """
        Create and save test metadata CSV file.

        Args:
            file_path: Path where to save the CSV
            num_samples: Number of samples to create
            class_balance: Ratio of positive class

        Returns:
            Created DataFrame
        """
        np.random.seed(42)  # For reproducible tests

        patient_ids = [f"patient_{i:04d}" for i in range(num_samples)]
        num_positive = int(num_samples * class_balance)
        targets = [1] * num_positive + [0] * (num_samples - num_positive)
        np.random.shuffle(targets)

        df = pd.DataFrame(
            {
                "patientId": patient_ids,
                "Target": targets,
                "age": np.random.randint(20, 90, num_samples),
                "gender": np.random.choice(["M", "F"], num_samples),
            },
        )

        df.to_csv(file_path, index=False)
        return df

    @staticmethod
    def verify_file_structure(
        base_path: Union[str, Path],
        expected_structure: Dict,
    ) -> None:
        """
        Verify that file system structure matches expected structure.

        Args:
            base_path: Base path to check
            expected_structure: Dictionary describing expected structure
        """
        base_path = Path(base_path)

        for item_name, item_type in expected_structure.items():
            item_path = base_path / item_name

            if item_type == "file":
                assert item_path.is_file(), f"Expected file not found: {item_path}"
            elif item_type == "dir":
                assert item_path.is_dir(), f"Expected directory not found: {item_path}"
            elif isinstance(item_type, dict):
                assert item_path.is_dir(), f"Expected directory not found: {item_path}"
                # Recursively check subdirectories
                TestHelpers.verify_file_structure(item_path, item_type)

    @staticmethod
    def create_minimal_test_environment(
        base_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """
        Create minimal test environment with required files and directories.

        Args:
            base_dir: Base directory (uses temp dir if None)

        Returns:
            Dictionary with paths to created components
        """
        if base_dir is None:
            base_dir = Path(tempfile.mkdtemp())
        else:
            base_dir = Path(base_dir)

        # Create directory structure
        images_dir = base_dir / "Images" / "Images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal metadata
        metadata_df = pd.DataFrame(
            {
                "patientId": [
                    "patient_001",
                    "patient_002",
                    "patient_003",
                    "patient_004",
                ],
                "Target": [0, 1, 0, 1],
            },
        )

        metadata_path = base_dir / "Train_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)

        # Create empty image files
        for patient_id in metadata_df["patientId"]:
            image_file = images_dir / f"{patient_id}.png"
            image_file.touch()

        return {
            "base_path": base_dir,
            "metadata_path": metadata_path,
            "images_dir": images_dir,
            "metadata_df": metadata_df,
        }

    @staticmethod
    def compare_configs(
        config1: Union[SystemConstants, ExperimentConfig],
        config2: Union[SystemConstants, ExperimentConfig],
    ) -> bool:
        """
        Compare two configuration objects for equality.

        Args:
            config1: First configuration
            config2: Second configuration

        Returns:
            True if configurations are equal
        """
        if not isinstance(config1, type(config2)) or not isinstance(
            config2,
            type(config1),
        ):
            return False

        if hasattr(config1, "to_dict"):
            return config1.to_dict() == config2.to_dict()
        else:
            # For dataclasses like SystemConstants
            return config1 == config2

    @staticmethod
    def mock_successful_training_run() -> Dict:
        """Create mock data simulating successful training run."""
        return {
            "train_metrics": {
                "accuracy": 0.85,
                "loss": 0.45,
                "precision": 0.83,
                "recall": 0.87,
                "f1": 0.85,
            },
            "val_metrics": {
                "accuracy": 0.82,
                "loss": 0.52,
                "precision": 0.80,
                "recall": 0.84,
                "f1": 0.82,
            },
            "epochs_completed": 10,
            "best_epoch": 8,
            "training_time": 120.5,
        }

    @staticmethod
    def mock_federated_learning_run(num_clients: int = 3, num_rounds: int = 5) -> Dict:
        """Create mock data simulating federated learning run."""
        return {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "rounds_completed": num_rounds,
            "client_metrics": {
                f"client_{i}": {
                    "samples": np.random.randint(50, 200),
                    "accuracy": 0.75 + np.random.random() * 0.15,
                    "loss": 0.3 + np.random.random() * 0.4,
                }
                for i in range(num_clients)
            },
            "global_metrics": {"accuracy": 0.83, "loss": 0.48},
            "aggregation_time": 15.2,
        }


class MockComponents:
    """Factory for creating mock components."""

    @staticmethod
    def create_mock_data_processor() -> Mock:
        """Create mock DataProcessor."""
        mock_processor = Mock()
        mock_processor.load_and_process_data = Mock(
            return_value=(
                pd.DataFrame(
                    {
                        "patientId": ["001", "002"],
                        "Target": [0, 1],
                        "filename": ["001.png", "002.png"],
                    },
                ),
                pd.DataFrame(
                    {"patientId": ["003"], "Target": [1], "filename": ["003.png"]},
                ),
            ),
        )
        mock_processor.validate_image_paths = Mock(return_value=True)
        return mock_processor

    @staticmethod
    def create_mock_config_loader() -> Mock:
        """Create mock ConfigLoader."""
        mock_loader = Mock()
        mock_loader.load_config = Mock(return_value={"system": {"batch_size": 32}})
        mock_loader.create_system_constants = Mock(return_value=SystemConstants())
        mock_loader.create_experiment_config = Mock(return_value=ExperimentConfig())
        return mock_loader

    @staticmethod
    def create_mock_model() -> Mock:
        """Create mock ML model."""
        mock_model = Mock()
        mock_model.forward = Mock(return_value=Mock())
        mock_model.train = Mock()
        mock_model.eval = Mock()
        mock_model.state_dict = Mock(return_value={})
        mock_model.load_state_dict = Mock()
        return mock_model


# Utility decorators for tests
def requires_gpu(func):
    """Decorator to skip tests that require GPU if not available."""
    import pytest
    import torch

    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        return func(*args, **kwargs)

    return wrapper


def slow_test(func):
    """Decorator to mark tests as slow."""
    import pytest

    return pytest.mark.slow(func)


def phase_test(phase_number: int):
    """Decorator to mark tests by development phase."""
    import pytest

    def decorator(func):
        return pytest.mark.__getattr__(f"phase{phase_number}")(func)

    return decorator
