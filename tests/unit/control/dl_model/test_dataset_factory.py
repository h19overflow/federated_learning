from unittest.mock import MagicMock

import pytest

from federated_pneumonia_detection.src.control.dl_model.internals.model.data_module_utils.dataset_factory import (  # noqa: E501
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
