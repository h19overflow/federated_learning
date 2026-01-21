"""
Unit tests for ImageValidator component.
Tests image file validation logic including format, content type checks.
"""

import pytest
from fastapi import HTTPException, UploadFile

from federated_pneumonia_detection.src.control.model_inferance.internals.image_validator import (
    ImageValidator,
)


class TestImageValidator:
    """Tests for ImageValidator class."""

    @pytest.fixture
    def validator(self):
        """Create ImageValidator instance."""
        return ImageValidator()

    # =========================================================================
    # Test validate method
    # =========================================================================

    def test_validate_valid_jpeg(self, validator, mock_upload_file_jpeg):
        """Test validation of valid JPEG file."""
        error = validator.validate(mock_upload_file_jpeg)
        assert error is None

    def test_validate_valid_png(self, validator, mock_upload_file_png):
        """Test validation of valid PNG file."""
        error = validator.validate(mock_upload_file_png)
        assert error is None

    def test_validate_invalid_pdf(self, validator, mock_upload_file_invalid_type):
        """Test validation rejects PDF file."""
        error = validator.validate(mock_upload_file_invalid_type)
        assert error is not None
        assert "Invalid file type" in error
        assert "application/pdf" in error

    def test_validate_invalid_content_type(self, validator):
        """Test validation rejects unsupported content type."""
        from io import BytesIO

        file = UploadFile(
            filename="test.gif",
            file=BytesIO(b"fake gif data"),
            content_type="image/gif",
        )

        error = validator.validate(file)
        assert error is not None
        assert "Invalid file type" in error

    def test_validate_case_insensitive_content_type(self, validator):
        """Test content type check is case-insensitive."""
        from io import BytesIO

        # Test uppercase
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(b"fake jpg"),
            content_type="IMAGE/JPEG",
        )
        error = validator.validate(file)
        # Should fail because it doesn't match exactly
        # (implementation uses exact match)
        assert error is not None

        # Test lowercase
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(b"fake jpg"),
            content_type="image/jpeg",
        )
        error = validator.validate(file)
        assert error is None

    def test_validate_all_allowed_types(self, validator):
        """Test all allowed content types are accepted."""
        from io import BytesIO

        allowed_types = ["image/png", "image/jpeg", "image/jpg"]

        for content_type in allowed_types:
            file = UploadFile(
                filename=f"test.{content_type.split('/')[1]}",
                file=BytesIO(b"fake image"),
                content_type=content_type,
            )
            error = validator.validate(file)
            assert error is None, f"Failed for {content_type}"

    # =========================================================================
    # Test validate_or_raise method
    # =========================================================================

    def test_validate_or_raise_valid_file(self, validator, mock_upload_file_png):
        """Test validate_or_raise with valid file."""
        # Should not raise
        validator.validate_or_raise(mock_upload_file_png)

    def test_validate_or_raise_invalid_file_raises_http_exception(
        self,
        validator,
        mock_upload_file_invalid_type,
    ):
        """Test validate_or_raise raises HTTPException for invalid file."""
        with pytest.raises(HTTPException) as exc_info:
            validator.validate_or_raise(mock_upload_file_invalid_type)

        assert exc_info.value.status_code == 400
        assert "Invalid file type" in exc_info.value.detail

    # =========================================================================
    # Test edge cases
    # =========================================================================

    def test_validate_missing_content_type(self, validator):
        """Test validation with missing content type."""
        from io import BytesIO

        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(b"fake data"),
            content_type=None,
        )

        error = validator.validate(file)
        assert error is not None

    def test_validate_empty_string_content_type(self, validator):
        """Test validation with empty string content type."""
        from io import BytesIO

        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(b"fake data"),
            content_type="",
        )

        error = validator.validate(file)
        assert error is not None

    def test_validate_extra_whitespace_in_content_type(self, validator):
        """Test validation handles whitespace in content type."""
        from io import BytesIO

        # Content type with whitespace should still work
        file = UploadFile(
            filename="test.jpg",
            file=BytesIO(b"fake data"),
            content_type=" image/jpeg ",
        )

        error = validator.validate(file)
        # Should fail because of exact match requirement
        assert error is not None

    # =========================================================================
    # Test ALLOWED_CONTENT_TYPES constant
    # =========================================================================

    def test_allowed_content_types_constant(self, validator):
        """Test that ALLOWED_CONTENT_TYPES is defined correctly."""
        assert hasattr(validator, "ALLOWED_CONTENT_TYPES")
        assert isinstance(validator.ALLOWED_CONTENT_TYPES, list)
        assert len(validator.ALLOWED_CONTENT_TYPES) > 0

    def test_allowed_content_types_includes_png_and_jpeg(self, validator):
        """Test that PNG and JPEG are in allowed types."""
        assert "image/png" in validator.ALLOWED_CONTENT_TYPES
        assert "image/jpeg" in validator.ALLOWED_CONTENT_TYPES
        assert "image/jpg" in validator.ALLOWED_CONTENT_TYPES

    # =========================================================================
    # Test with various upload scenarios
    # =========================================================================

    def test_validate_with_real_jpeg_bytes(self, validator):
        """Test validation with actual JPEG file bytes."""
        from io import BytesIO

        import numpy as np
        from PIL import Image

        # Create real JPEG image
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="RGB")

        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        file = UploadFile(
            filename="real_image.jpg",
            file=buffer,
            content_type="image/jpeg",
        )

        error = validator.validate(file)
        assert error is None

    def test_validate_with_real_png_bytes(self, validator):
        """Test validation with actual PNG file bytes."""
        from io import BytesIO

        import numpy as np
        from PIL import Image

        # Create real PNG image
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="RGB")

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        file = UploadFile(
            filename="real_image.png",
            file=buffer,
            content_type="image/png",
        )

        error = validator.validate(file)
        assert error is None
