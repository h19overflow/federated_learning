import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
import logging

from federated_pneumonia_detection.src.control.dl_model.internals.model.data_module_utils.dataset_factory import (
    create_dataset,
    create_training_transforms,
    create_validation_transforms,
)


@pytest.fixture
def mock_transform_builder():
    """Mock TransformBuilder for testing."""
    builder = MagicMock()
    builder.augmentation_strength = 0.5
    builder.build_training_transforms.return_value = MagicMock(
        name="training_transforms"
    )
    builder.build_validation_transforms.return_value = MagicMock(
        name="validation_transforms"
    )
    return builder


class TestDatasetFactory:
    """Test suite for dataset_factory utilities."""

    @patch(
        "federated_pneumonia_detection.src.control.dl_model.internals.model.data_module_utils.dataset_factory.CustomImageDataset"
    )
    def test_create_dataset_success(
        self,
        mock_custom_dataset_class,
        sample_dataframe,
        mock_config,
        mock_transform,
        caplog,
    ):
        """Test successful dataset creation and logging."""
        # Setup mock dataset instance
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 10
        mock_dataset_instance.get_class_distribution.return_value = {"0": 5, "1": 5}
        mock_dataset_instance.get_memory_usage_estimate.return_value = {
            "estimated_total_memory_mb": 100.5
        }
        mock_custom_dataset_class.return_value = mock_dataset_instance

        image_dir = Path("/tmp/images")
        dataset_type = "train"

        with caplog.at_level(logging.INFO):
            dataset = create_dataset(
                dataframe=sample_dataframe,
                image_dir=image_dir,
                config=mock_config,
                transforms=mock_transform,
                color_mode="RGB",
                validate_images=True,
                dataset_type=dataset_type,
            )

        # Verify initialization
        mock_custom_dataset_class.assert_called_once_with(
            dataframe=sample_dataframe,
            image_dir=image_dir,
            config=mock_config,
            transform=mock_transform,
            color_mode="RGB",
            validate_images=True,
        )

        # Verify return value
        assert dataset == mock_dataset_instance

        # Verify logging
        assert f"{dataset_type.capitalize()} dataset created" in caplog.text
        assert "10 samples" in caplog.text
        assert "classes: {'0': 5, '1': 5}" in caplog.text
        assert "estimated memory: 100.5 MB" in caplog.text

    @patch(
        "federated_pneumonia_detection.src.control.dl_model.internals.model.data_module_utils.dataset_factory.CustomImageDataset"
    )
    def test_create_dataset_empty_warning(
        self,
        mock_custom_dataset_class,
        sample_dataframe,
        mock_config,
        mock_transform,
        caplog,
    ):
        """Test warning when dataset is empty."""
        # Setup mock dataset instance as empty
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 0
        mock_custom_dataset_class.return_value = mock_dataset_instance

        dataset_type = "val"

        with caplog.at_level(logging.WARNING):
            dataset = create_dataset(
                dataframe=sample_dataframe,
                image_dir=Path("/tmp/images"),
                config=mock_config,
                transforms=mock_transform,
                color_mode="RGB",
                validate_images=False,
                dataset_type=dataset_type,
            )

        assert dataset == mock_dataset_instance
        assert f"{dataset_type.capitalize()} dataset is empty" in caplog.text

    def test_create_training_transforms(self, mock_transform_builder):
        """Test creation of training transforms with augmentation enabled."""
        custom_config = {"resize": [224, 224]}

        transforms = create_training_transforms(
            transform_builder=mock_transform_builder,
            custom_preprocessing_config=custom_config,
        )

        mock_transform_builder.build_training_transforms.assert_called_once_with(
            enable_augmentation=True,
            augmentation_strength=mock_transform_builder.augmentation_strength,
            custom_preprocessing=custom_config,
        )
        assert (
            transforms == mock_transform_builder.build_training_transforms.return_value
        )

    def test_create_training_transforms_no_custom_config(self, mock_transform_builder):
        """Test training transforms without custom config."""
        create_training_transforms(
            transform_builder=mock_transform_builder, custom_preprocessing_config=None
        )

        mock_transform_builder.build_training_transforms.assert_called_once_with(
            enable_augmentation=True,
            augmentation_strength=mock_transform_builder.augmentation_strength,
            custom_preprocessing=None,
        )

    def test_create_validation_transforms(self, mock_transform_builder):
        """Test creation of validation transforms."""
        custom_config = {"resize": [224, 224]}

        transforms = create_validation_transforms(
            transform_builder=mock_transform_builder,
            custom_preprocessing_config=custom_config,
        )

        mock_transform_builder.build_validation_transforms.assert_called_once_with(
            custom_preprocessing=custom_config
        )
        assert (
            transforms
            == mock_transform_builder.build_validation_transforms.return_value
        )

    def test_create_validation_transforms_no_custom_config(
        self, mock_transform_builder
    ):
        """Test validation transforms without custom config."""
        create_validation_transforms(
            transform_builder=mock_transform_builder, custom_preprocessing_config=None
        )

        mock_transform_builder.build_validation_transforms.assert_called_once_with(
            custom_preprocessing=None
        )
