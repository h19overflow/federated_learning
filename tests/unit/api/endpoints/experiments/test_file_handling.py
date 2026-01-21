"""
Unit tests for file handling utilities in experiment endpoints.

Tests cover:
- ZIP file extraction
- Root directory detection
- Error handling for invalid files
- Temporary directory cleanup
"""

import os
import tempfile
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from federated_pneumonia_detection.src.api.endpoints.experiments.utils.file_handling import (
    prepare_zip,
)


class TestPrepareZip:
    """Test prepare_zip function for handling uploaded ZIP files."""

    async def test_prepare_zip_valid_zip(
        self,
        sample_zip_file,
        experiment_logger,
        tmp_path,
    ):
        """Test successful ZIP extraction with valid structure."""
        # Mock UploadFile
        mock_upload = MagicMock()
        mock_upload.filename = "test_data.zip"

        with open(sample_zip_file, "rb") as f:
            mock_upload.read = MagicMock(return_value=f.read())
            mock_upload.seek = MagicMock(side_effect=lambda x: f.seek(x))

        source_path = await prepare_zip(mock_upload, experiment_logger, "test_exp")

        # Verify extraction
        assert os.path.exists(source_path)
        assert os.path.exists(os.path.join(source_path, "Images"))
        assert os.path.exists(os.path.join(source_path, "stage2_train_metadata.csv"))

    async def test_prepare_zip_with_root_directory(
        self,
        sample_zip_with_root_dir,
        experiment_logger,
    ):
        """Test ZIP with single root directory wrapper."""
        mock_upload = MagicMock()
        mock_upload.filename = "test_wrapped.zip"

        with open(sample_zip_with_root_dir, "rb") as f:
            mock_upload.read = MagicMock(return_value=f.read())
            mock_upload.seek = MagicMock(side_effect=lambda x: f.seek(x))

        source_path = await prepare_zip(mock_upload, experiment_logger, "test_exp")

        # Verify root directory was detected
        assert os.path.exists(source_path)
        assert os.path.exists(os.path.join(source_path, "Images"))
        assert os.path.exists(os.path.join(source_path, "metadata.csv"))

    async def test_prepare_zip_direct_structure(self, experiment_logger, tmp_path):
        """Test ZIP with direct structure (no root directory)."""
        # Create ZIP with direct structure
        zip_path = tmp_path / "direct.zip"

        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = os.path.join(temp_dir, "Images")
            os.makedirs(images_dir)

            # Create dummy images
            for i in range(2):
                with open(os.path.join(images_dir, f"img_{i}.jpg"), "wb") as f:
                    f.write(b"data")

            csv_path = os.path.join(temp_dir, "data.csv")
            with open(csv_path, "w") as f:
                f.write("file,label\nimg_0.jpg,0\n")

            with zipfile.ZipFile(zip_path, "w") as zf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zf.write(file_path, arcname)

        mock_upload = MagicMock()
        mock_upload.filename = "direct.zip"

        with open(zip_path, "rb") as f:
            mock_upload.read = MagicMock(return_value=f.read())
            mock_upload.seek = MagicMock(side_effect=lambda x: f.seek(x))

        source_path = await prepare_zip(mock_upload, experiment_logger, "test_exp")

        # Verify direct structure was used
        assert os.path.exists(source_path)
        assert os.path.exists(os.path.join(source_path, "Images"))

    async def test_prepare_zip_invalid_file(self, experiment_logger, invalid_zip_file):
        """Test error handling for invalid ZIP file."""
        mock_upload = MagicMock()
        mock_upload.filename = "invalid.zip"

        with open(invalid_zip_file, "rb") as f:
            mock_upload.read = MagicMock(return_value=f.read())

        with pytest.raises(Exception):
            await prepare_zip(mock_upload, experiment_logger, "test_exp")

    async def test_prepare_zip_temp_dir_cleanup_on_error(
        self,
        experiment_logger,
        invalid_zip_file,
    ):
        """Test that temporary directory is cleaned up on error."""
        mock_upload = MagicMock()
        mock_upload.filename = "invalid.zip"

        with open(invalid_zip_file, "rb") as f:
            mock_upload.read = MagicMock(return_value=f.read())

        # Track temp directories created
        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.file_handling.tempfile.mkdtemp",
        ) as mock_mkdtemp:
            temp_dir = tempfile.mkdtemp()
            mock_mkdtemp.return_value = temp_dir

            with pytest.raises(Exception):
                await prepare_zip(mock_upload, experiment_logger, "test_exp")

            # Verify cleanup was attempted
            assert not os.path.exists(temp_dir)

    async def test_prepare_zip_handles_multiple_items_in_root(
        self,
        experiment_logger,
        tmp_path,
    ):
        """Test ZIP with multiple items at root level."""
        zip_path = tmp_path / "multi_root.zip"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple items at root
            os.makedirs(os.path.join(temp_dir, "Images"))
            os.makedirs(os.path.join(temp_dir, "ExtraDir"))

            with open(os.path.join(temp_dir, "data.csv"), "w") as f:
                f.write("file,label\n")

            with zipfile.ZipFile(zip_path, "w") as zf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zf.write(file_path, arcname)

            # Add a directory (requires special handling)
            for root, dirs, _ in os.walk(temp_dir):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    arcname = os.path.relpath(dir_path, temp_dir)
                    zf.write(dir_path, arcname)

        mock_upload = MagicMock()
        mock_upload.filename = "multi_root.zip"

        with open(zip_path, "rb") as f:
            mock_upload.read = MagicMock(return_value=f.read())
            mock_upload.seek = MagicMock(side_effect=lambda x: f.seek(x))

        source_path = await prepare_zip(mock_upload, experiment_logger, "test_exp")

        # Should return extraction path directly
        assert os.path.exists(source_path)

    async def test_prepare_zip_single_non_root_directory(
        self,
        experiment_logger,
        tmp_path,
    ):
        """Test ZIP with single directory that's not a valid root."""
        zip_path = tmp_path / "single_non_root.zip"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create single directory without Images/ or CSV
            single_dir = os.path.join(temp_dir, "NotARoot")
            os.makedirs(single_dir)

            with open(os.path.join(single_dir, "other.txt"), "w") as f:
                f.write("content")

            with zipfile.ZipFile(zip_path, "w") as zf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zf.write(file_path, arcname)

        mock_upload = MagicMock()
        mock_upload.filename = "single_non_root.zip"

        with open(zip_path, "rb") as f:
            mock_upload.read = MagicMock(return_value=f.read())
            mock_upload.seek = MagicMock(side_effect=lambda x: f.seek(x))

        source_path = await prepare_zip(mock_upload, experiment_logger, "test_exp")

        # Should return extraction path (not the single dir)
        assert os.path.exists(source_path)
