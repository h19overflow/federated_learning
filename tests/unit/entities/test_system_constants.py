"""
Unit tests for SystemConstants entity class.
Tests configuration values, validation, and custom creation methods.
"""

import pytest

from federated_pneumonia_detection.models.system_constants import SystemConstants


class TestSystemConstants:
    """Test cases for SystemConstants entity."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        constants = SystemConstants()

        assert constants.IMG_SIZE == (224, 224)
        assert constants.IMAGE_EXTENSION == ".png"
        assert constants.BATCH_SIZE == 128
        assert constants.SAMPLE_FRACTION == 0.05
        assert constants.VALIDATION_SPLIT == 0.20
        assert constants.SEED == 42
        assert constants.BASE_PATH == "."
        assert constants.MAIN_IMAGES_FOLDER == "Images"
        assert constants.IMAGES_SUBFOLDER == "Images"
        assert constants.METADATA_FILENAME == "Train_metadata.csv"
        assert constants.PATIENT_ID_COLUMN == "patientId"
        assert constants.TARGET_COLUMN == "Target"
        assert constants.FILENAME_COLUMN == "filename"

    def test_frozen_dataclass(self):
        """Test that SystemConstants is immutable."""
        constants = SystemConstants()

        with pytest.raises(AttributeError):
            constants.BATCH_SIZE = 256

    def test_create_custom_basic(self):
        """Test creating custom constants with basic parameters."""
        custom_constants = SystemConstants.create_custom(
            img_size=(256, 256),
            batch_size=64,
            seed=123,
        )

        assert custom_constants.IMG_SIZE == (256, 256)
        assert custom_constants.BATCH_SIZE == 64
        assert custom_constants.SEED == 123
        # Other values should use create_custom defaults (not dataclass defaults)
        assert custom_constants.SAMPLE_FRACTION == 0.10
        assert custom_constants.IMAGE_EXTENSION == ".png"

    def test_create_custom_all_parameters(self):
        """Test creating custom constants with all parameters."""
        custom_constants = SystemConstants.create_custom(
            img_size=(512, 512),
            batch_size=32,
            sample_fraction=0.50,
            validation_split=0.30,
            seed=999,
            base_path="/custom/path",
            main_images_folder="CustomImages",
            images_subfolder="XRays",
            metadata_filename="custom_metadata.csv",
            image_extension=".jpg",
        )

        assert custom_constants.IMG_SIZE == (512, 512)
        assert custom_constants.BATCH_SIZE == 32
        assert custom_constants.SAMPLE_FRACTION == 0.50
        assert custom_constants.VALIDATION_SPLIT == 0.30
        assert custom_constants.SEED == 999
        assert custom_constants.BASE_PATH == "/custom/path"
        assert custom_constants.MAIN_IMAGES_FOLDER == "CustomImages"
        assert custom_constants.IMAGES_SUBFOLDER == "XRays"
        assert custom_constants.METADATA_FILENAME == "custom_metadata.csv"
        assert custom_constants.IMAGE_EXTENSION == ".jpg"

    def test_img_size_type(self):
        """Test that IMG_SIZE is properly typed as tuple."""
        constants = SystemConstants()
        assert isinstance(constants.IMG_SIZE, tuple)
        assert len(constants.IMG_SIZE) == 2
        assert all(isinstance(dim, int) for dim in constants.IMG_SIZE)

    def test_string_attributes(self):
        """Test that string attributes are properly set."""
        constants = SystemConstants()

        string_attrs = [
            "IMAGE_EXTENSION",
            "BASE_PATH",
            "MAIN_IMAGES_FOLDER",
            "IMAGES_SUBFOLDER",
            "METADATA_FILENAME",
            "PATIENT_ID_COLUMN",
            "TARGET_COLUMN",
            "FILENAME_COLUMN",
        ]

        for attr in string_attrs:
            value = getattr(constants, attr)
            assert isinstance(value, str)
            assert len(value) > 0

    def test_numeric_attributes(self):
        """Test that numeric attributes have valid ranges."""
        constants = SystemConstants()

        assert constants.BATCH_SIZE > 0
        assert 0 < constants.SAMPLE_FRACTION <= 1
        assert 0 < constants.VALIDATION_SPLIT < 1
        assert constants.SEED >= 0

    def test_equality(self):
        """Test that two SystemConstants instances with same values are equal."""
        constants1 = SystemConstants()
        constants2 = SystemConstants()

        assert constants1 == constants2

    def test_custom_equality(self):
        """Test equality of custom constants."""
        custom1 = SystemConstants.create_custom(batch_size=64, seed=123)
        custom2 = SystemConstants.create_custom(batch_size=64, seed=123)
        custom3 = SystemConstants.create_custom(batch_size=128, seed=123)

        assert custom1 == custom2
        assert custom1 != custom3
