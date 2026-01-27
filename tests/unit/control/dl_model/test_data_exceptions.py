import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from federated_pneumonia_detection.src.control.dl_model.internals.data.xray_data_module import (  # noqa: E501
    XRayDataModule,
)
from federated_pneumonia_detection.src.control.dl_model.internals.model.data_module_utils.validation import (  # noqa: E501
    validate_inputs,
)


class TestDataExceptions:
    """Test suite for exception handling in data module and validation utilities."""

    # Tests for validation.py
    def test_validate_image_dir_missing(self):
        """Test that FileNotFoundError is raised when image directory does not exist."""
        image_dir = Path("/non/existent/path")
        expected_msg = re.escape(f"Image directory not found: {image_dir}")
        with pytest.raises(FileNotFoundError, match=expected_msg):
            validate_inputs(
                image_dir=image_dir,
                color_mode="RGB",
                train_df=pd.DataFrame(),
                val_df=pd.DataFrame(),
                filename_column="path",
                target_column="label",
            )

    def test_validate_image_dir_not_dir(self, tmp_path):
        """Test that ValueError is raised when image directory path is a file."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("dummy content")
        expected_msg = re.escape(
            f"Image directory path is not a directory: {file_path}"
        )
        with pytest.raises(ValueError, match=expected_msg):
            validate_inputs(
                image_dir=file_path,
                color_mode="RGB",
                train_df=pd.DataFrame(),
                val_df=pd.DataFrame(),
                filename_column="path",
                target_column="label",
            )

    def test_validate_color_mode_invalid(self, tmp_path):
        """Test that ValueError is raised for invalid color mode."""
        image_dir = tmp_path
        with pytest.raises(ValueError, match="color_mode must be 'RGB' or 'L'"):
            validate_inputs(
                image_dir=image_dir,
                color_mode="CMYK",
                train_df=pd.DataFrame({"path": ["1.jpg"], "label": [0]}),
                val_df=pd.DataFrame({"path": ["2.jpg"], "label": [1]}),
                filename_column="path",
                target_column="label",
            )

    def test_validate_dataframe_empty(self, tmp_path):
        """Test that ValueError is raised when both dataframes are empty."""
        image_dir = tmp_path
        with pytest.raises(
            ValueError, match="Both train and validation DataFrames are empty"
        ):
            validate_inputs(
                image_dir=image_dir,
                color_mode="RGB",
                train_df=pd.DataFrame(),
                val_df=pd.DataFrame(),
                filename_column="path",
                target_column="label",
            )

    def test_validate_dataframe_missing_cols(self, tmp_path):
        """Test that ValueError is raised when required columns are missing."""
        image_dir = tmp_path
        train_df = pd.DataFrame({"wrong_col": [1]})
        val_df = pd.DataFrame({"path": ["2.jpg"], "label": [1]})

        # Using re.escape for the list representation which contains brackets
        expected_msg = re.escape(
            "Missing columns in train DataFrame: ['path', 'label']"
        )
        with pytest.raises(ValueError, match=expected_msg):
            validate_inputs(
                image_dir=image_dir,
                color_mode="RGB",
                train_df=train_df,
                val_df=val_df,
                filename_column="path",
                target_column="label",
            )

    # Tests for XRayDataModule
    def test_dataloader_before_setup(self, tmp_path):
        """Test that RuntimeError is raised when calling dataloader before setup."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        train_df = pd.DataFrame({"filename": ["1.jpg"], "Target": [0]})
        val_df = pd.DataFrame({"filename": ["2.jpg"], "Target": [1]})

        mock_config = MagicMock()
        # Mock config.get to return expected defaults or specific values
        config_values = {
            "experiment.color_mode": "RGB",
            "experiment.num_workers": 0,
            "experiment.validate_images_on_init": False,
            "columns.filename": "filename",
            "columns.target": "Target",
            "experiment.pin_memory": False,
            "experiment.persistent_workers": False,
            "experiment.prefetch_factor": 2,
        }
        mock_config.get.side_effect = lambda key, default=None: config_values.get(
            key, default
        )

        with patch(
            "federated_pneumonia_detection.src.control.dl_model.internals.data.xray_data_module.TransformBuilder"
        ):
            dm = XRayDataModule(
                train_df=train_df,
                val_df=val_df,
                image_dir=image_dir,
                config=mock_config,
            )

            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    "Training dataset not initialized. Call setup() first."
                ),
            ):
                dm.train_dataloader()

            with pytest.raises(
                RuntimeError,
                match=re.escape(
                    "Validation dataset not initialized. Call setup() first."
                ),
            ):
                dm.val_dataloader()
