"""
Unit tests for core/utils module.

Tests federated learning utility functions.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from federated_pneumonia_detection.src.control.federated_new_version.core.utils import (
    _convert_metric_record_to_dict,
    _create_metric_record_dict,
    _extract_metrics_from_result,
    _prepare_evaluation_dataframe,
    _prepare_partition_and_split,
    filter_list_of_dicts,
)


class TestFilterListOfDicts:
    """Test suite for filter_list_of_dicts function."""

    def test_filter_single_field(self):
        """Test filtering list with single field from last epoch."""
        data = [{"epoch": 1, "train_loss": 0.5, "train_acc": 0.8}]
        fields = ["epoch"]
        result = filter_list_of_dicts(data, fields)

        assert result == {"epoch": 1}

    def test_filter_multiple_fields(self):
        """Test filtering list with multiple fields from last epoch."""
        data = [
            {"epoch": 1, "train_loss": 0.5, "train_acc": 0.8},
            {"epoch": 2, "train_loss": 0.4, "train_acc": 0.85},
        ]
        fields = ["epoch", "train_loss"]
        result = filter_list_of_dicts(data, fields)

        # Should return last epoch values
        assert result == {"epoch": 2, "train_loss": 0.4}

    def test_filter_empty_list_returns_defaults(self):
        """Test filtering empty list returns defaults (0.0) for all fields."""
        data = []
        fields = ["epoch", "train_loss", "train_acc"]
        result = filter_list_of_dicts(data, fields)

        # Should return defaults (0.0) for all requested fields
        assert result == {"epoch": 0.0, "train_loss": 0.0, "train_acc": 0.0}

    def test_filter_missing_fields_return_defaults(self):
        """Test that missing fields return default (0.0) values."""
        data = [{"epoch": 1, "train_loss": 0.5}]
        fields = ["epoch", "train_loss", "train_acc", "val_f1"]
        result = filter_list_of_dicts(data, fields)

        # train_acc and val_f1 should have default 0.0
        assert result["epoch"] == 1
        assert result["train_loss"] == 0.5
        assert result["train_acc"] == 0.0  # Default
        assert result["val_f1"] == 0.0  # Default

    def test_filter_all_fields(self):
        """Test filtering all fields from last epoch."""
        data = [{"epoch": 1, "train_loss": 0.5, "train_acc": 0.8}]
        fields = ["epoch", "train_loss", "train_acc"]
        result = filter_list_of_dicts(data, fields)

        assert result == {"epoch": 1, "train_loss": 0.5, "train_acc": 0.8}

    def test_filter_uses_last_epoch_only(self):
        """Test that only last epoch is used, not all epochs."""
        data = [
            {"epoch": 1, "train_loss": 0.5},
            {"epoch": 2, "train_loss": 0.4},
            {"epoch": 3, "train_loss": 0.3},
        ]
        fields = ["epoch", "train_loss"]
        result = filter_list_of_dicts(data, fields)

        # Should have last epoch values (epoch 3)
        assert result["epoch"] == 3
        assert result["train_loss"] == 0.3

    def test_filter_preserves_zero_values(self):
        """Test that legitimate zero values are preserved, not replaced with defaults."""
        data = [{"epoch": 1, "train_loss": 0.0, "train_acc": 0.0}]
        fields = ["epoch", "train_loss", "train_acc", "val_f1"]
        result = filter_list_of_dicts(data, fields)

        # Zero values should be preserved, val_f1 should be default 0.0
        assert result["epoch"] == 1
        assert result["train_loss"] == 0.0
        assert result["train_acc"] == 0.0
        assert result["val_f1"] == 0.0  # Default (not in data)


class TestExtractMetricsFromResult:
    """Test suite for _extract_metrics_from_result function."""

    def test_extract_all_metrics_present(self):
        """Test extracting all metrics when all are present."""
        result_dict = {
            "test_loss": 0.5,
            "test_accuracy": 0.8,
            "test_precision": 0.75,
            "test_recall": 0.7,
            "test_f1": 0.72,
            "test_auroc": 0.85,
        }

        loss, acc, prec, rec, f1, auroc, tp, tn, fp, fn = _extract_metrics_from_result(
            result_dict
        )

        assert loss == 0.5
        assert acc == 0.8
        assert prec == 0.75
        assert rec == 0.7
        assert f1 == 0.72
        assert auroc == 0.85

    def test_extract_with_alternative_names(self):
        """Test extracting metrics with alternative naming."""
        result_dict = {
            "loss": 0.5,
            "test_acc": 0.8,  # Alternative to test_accuracy
            "precision": 0.75,  # Alternative to test_precision
            "recall": 0.7,
            "f1": 0.72,
            "auroc": 0.85,
        }

        loss, acc, prec, rec, f1, auroc, tp, tn, fp, fn = _extract_metrics_from_result(
            result_dict
        )

        assert loss == 0.5
        assert acc == 0.8
        assert prec == 0.75
        assert rec == 0.7
        assert f1 == 0.72
        assert auroc == 0.85

    def test_extract_with_zero_values(self):
        """Test extracting metrics when values are zero."""
        result_dict = {
            "test_loss": 0.0,
            "test_accuracy": 0.0,
            "test_precision": 0.0,
            "test_recall": 0.0,
            "test_f1": 0.0,
            "test_auroc": 0.0,
        }

        loss, acc, prec, rec, f1, auroc, tp, tn, fp, fn = _extract_metrics_from_result(
            result_dict
        )

        # Zero should be returned, not None or default
        assert loss == 0.0
        assert acc == 0.0
        assert prec == 0.0
        assert rec == 0.0
        assert f1 == 0.0
        assert auroc == 0.0

    def test_extract_with_missing_metrics(self):
        """Test extracting metrics when some are missing."""
        result_dict = {
            "test_loss": 0.5,
            "test_accuracy": 0.8,
        }

        loss, acc, prec, rec, f1, auroc, tp, tn, fp, fn = _extract_metrics_from_result(
            result_dict
        )

        assert loss == 0.5
        assert acc == 0.8
        assert prec == 0.0  # Default
        assert rec == 0.0  # Default
        assert f1 == 0.0  # Default
        assert auroc == 0.0  # Default

    def test_extract_with_none_values(self):
        """Test extracting metrics when some are None."""
        result_dict = {
            "test_loss": 0.5,
            "test_accuracy": None,
            "test_precision": 0.75,
        }

        loss, acc, prec, rec, f1, auroc, tp, tn, fp, fn = _extract_metrics_from_result(
            result_dict
        )

        assert loss == 0.5
        assert acc is None  # None preserved
        assert prec == 0.75
        assert rec == 0.0
        assert f1 == 0.0
        assert auroc == 0.0

    def test_extract_test_acc_over_test_accuracy(self):
        """Test that test_acc is preferred over test_accuracy."""
        result_dict = {
            "test_loss": 0.5,
            "test_acc": 0.8,
            "test_accuracy": 0.75,  # Should be ignored
        }

        loss, acc, prec, rec, f1, auroc, tp, tn, fp, fn = _extract_metrics_from_result(
            result_dict
        )

        assert loss == 0.5
        assert acc == 0.8  # test_acc used, not test_accuracy

    def test_extract_empty_dict(self):
        """Test extracting from empty dictionary."""
        result_dict = {}

        loss, acc, prec, rec, f1, auroc, tp, tn, fp, fn = _extract_metrics_from_result(
            result_dict
        )

        assert loss == 0.0
        assert acc == 0.0
        assert prec == 0.0
        assert rec == 0.0
        assert f1 == 0.0
        assert auroc == 0.0


class TestCreateMetricRecordDict:
    """Test suite for _create_metric_record_dict function."""

    def test_create_metric_record(self):
        """Test creating metric record dict."""
        result = _create_metric_record_dict(
            loss=0.5,
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1=0.72,
            auroc=0.85,
            num_examples=100,
        )

        assert result["test_loss"] == 0.5
        assert result["test_accuracy"] == 0.8
        assert result["test_precision"] == 0.75
        assert result["test_recall"] == 0.7
        assert result["test_f1"] == 0.72
        assert result["test_auroc"] == 0.85
        assert result["num-examples"] == 100  # Note: hyphen, not underscore

    def test_num_examples_key_format(self):
        """Test that num_examples uses hyphen format."""
        result = _create_metric_record_dict(
            loss=0.5,
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1=0.72,
            auroc=0.85,
            num_examples=50,
        )

        assert "num-examples" in result
        assert result["num-examples"] == 50
        # Should NOT have underscore version
        assert "num_examples" not in result

    def test_create_with_float_metrics(self):
        """Test creating with float metrics."""
        result = _create_metric_record_dict(
            loss=0.523456,
            accuracy=0.876543,
            precision=0.754321,
            recall=0.798765,
            f1=0.724567,
            auroc=0.856789,
            num_examples=100,
        )

        # Values should be preserved as floats
        assert isinstance(result["test_loss"], float)
        assert isinstance(result["test_accuracy"], float)
        assert result["test_loss"] == pytest.approx(0.523456)

    def test_create_with_zero_num_examples(self):
        """Test creating with zero num_examples."""
        result = _create_metric_record_dict(
            loss=0.5,
            accuracy=0.8,
            precision=0.75,
            recall=0.7,
            f1=0.72,
            auroc=0.85,
            num_examples=0,
        )

        assert result["num-examples"] == 0


class TestConvertMetricRecordToDict:
    """Test suite for _convert_metric_record_to_dict function."""

    def test_convert_simple_dict(self):
        """Test converting simple dictionary."""
        data = {"key1": "value1", "key2": 42, "key3": True}
        result = _convert_metric_record_to_dict(data)

        assert result == data

    def test_convert_nested_dict(self):
        """Test converting nested dictionary."""
        data = {"outer": {"inner": "value", "number": 10}}
        result = _convert_metric_record_to_dict(data)

        assert result["outer"]["inner"] == "value"
        assert result["outer"]["number"] == 10

    def test_convert_list(self):
        """Test converting list."""
        data = [1, 2, "three", {"key": "value"}]
        result = _convert_metric_record_to_dict(data)

        assert result == [1, 2, "three", {"key": "value"}]

    def test_convert_nested_list(self):
        """Test converting nested list."""
        data = {"list_key": [1, 2, {"nested": "value"}]}
        result = _convert_metric_record_to_dict(data)

        assert result["list_key"][0] == 1
        assert result["list_key"][2]["nested"] == "value"

    def test_convert_primitive_types(self):
        """Test converting primitive types."""
        data = {
            "int": 42,
            "float": 3.14,
            "string": "test",
            "bool": True,
            "none": None,
        }
        result = _convert_metric_record_to_dict(data)

        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["string"] == "test"
        assert result["bool"] is True
        assert result["none"] is None

    def test_convert_non_primitive_to_string(self):
        """Test that non-primitive types are converted to string."""
        obj = object()
        data = {"object_key": obj}
        result = _convert_metric_record_to_dict(data)

        assert isinstance(result["object_key"], str)
        assert result["object_key"] == str(obj)

    def test_convert_complex_nested_structure(self):
        """Test converting complex nested structure."""
        data = {
            "level1": {
                "level2": {
                    "level3": [1, 2, {"deep": "value"}],
                },
                "list": ["a", "b", {"c": "d"}],
            },
        }
        result = _convert_metric_record_to_dict(data)

        assert result["level1"]["level2"]["level3"][2]["deep"] == "value"
        assert result["level1"]["list"][2]["c"] == "d"

    def test_convert_dict_keys_to_strings(self):
        """Test that dictionary keys are converted to strings."""
        data = {1: "value1", 2.5: "value2"}
        result = _convert_metric_record_to_dict(data)

        assert "1" in result
        assert "2.5" in result
        assert result["1"] == "value1"


class TestPreparePartitionAndSplit:
    """Test suite for _prepare_partition_and_split function."""

    @patch(
        "federated_pneumonia_detection.src.control.federated_new_version.core.utils.train_test_split",
    )
    def test_partition_split_with_seed(self, mock_split):
        """Test partition split with specified seed."""
        mock_partitioner = Mock()
        mock_split.return_value = (
            pd.DataFrame({"a": [1, 2]}),
            pd.DataFrame({"b": [3]}),
        )

        train_df, val_df = _prepare_partition_and_split(
            mock_partitioner,
            0,
            pd.DataFrame(),
            seed=42,
        )

        mock_split.assert_called_once()
        call_kwargs = mock_split.call_args[1]
        assert call_kwargs["random_state"] == 42
        assert call_kwargs["test_size"] == 0.2

    @patch(
        "federated_pneumonia_detection.src.control.federated_new_version.core.utils.train_test_split",
    )
    def test_partition_split_default_seed(self, mock_split):
        """Test partition split with default seed."""
        mock_partitioner = Mock()
        mock_split.return_value = (
            pd.DataFrame({"a": [1, 2]}),
            pd.DataFrame({"b": [3]}),
        )

        train_df, val_df = _prepare_partition_and_split(
            mock_partitioner,
            0,
            pd.DataFrame(),
        )

        call_kwargs = mock_split.call_args[1]
        assert call_kwargs["random_state"] == 42  # Default

    @patch(
        "federated_pneumonia_detection.src.control.federated_new_version.core.utils.train_test_split",
    )
    def test_partition_split_correct_test_size(self, mock_split):
        """Test that test_size is set to 0.2."""
        mock_partitioner = Mock()
        mock_split.return_value = (
            pd.DataFrame({"a": [1, 2]}),
            pd.DataFrame({"b": [3]}),
        )

        train_df, val_df = _prepare_partition_and_split(
            mock_partitioner,
            0,
            pd.DataFrame(),
        )

        call_kwargs = mock_split.call_args[1]
        assert call_kwargs["test_size"] == 0.2

    @patch(
        "federated_pneumonia_detection.src.control.federated_new_version.core.utils.train_test_split",
    )
    def test_partition_split_returns_dataframes(self, mock_split):
        """Test that function returns DataFrames."""
        mock_partitioner = Mock()
        mock_split.return_value = (
            pd.DataFrame({"train": [1, 2, 3]}),
            pd.DataFrame({"val": [4]}),
        )

        train_df, val_df = _prepare_partition_and_split(
            mock_partitioner,
            0,
            pd.DataFrame(),
        )

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)


class TestPrepareEvaluationDataframe:
    """Test suite for _prepare_evaluation_dataframe function."""

    def test_adds_filename_column(self):
        """Test that filename column is added when missing."""
        df = pd.DataFrame(
            {
                "patientId": [1, 2, 3],
                "class": ["Normal", "Pneumonia", "Normal"],
            },
        )

        result = _prepare_evaluation_dataframe(df)

        assert "filename" in result.columns
        assert all(result["filename"].str.endswith(".png"))
        assert list(result["filename"]) == ["1.png", "2.png", "3.png"]

    def test_preserves_existing_filename(self):
        """Test that existing filename column is preserved."""
        df = pd.DataFrame(
            {
                "patientId": [1, 2, 3],
                "filename": ["custom1.png", "custom2.png", "custom3.png"],
                "class": ["Normal", "Pneumonia", "Normal"],
            },
        )

        result = _prepare_evaluation_dataframe(df)

        assert "filename" in result.columns
        assert list(result["filename"]) == ["custom1.png", "custom2.png", "custom3.png"]

    def test_returns_dataframe(self):
        """Test that function returns DataFrame."""
        df = pd.DataFrame({"patientId": [1, 2]})

        result = _prepare_evaluation_dataframe(df)

        assert isinstance(result, pd.DataFrame)

    def test_no_patientId_column(self):
        """Test handling when patientId column is missing."""
        df = pd.DataFrame({"class": ["Normal", "Pneumonia"]})

        # Should not add filename column
        result = _prepare_evaluation_dataframe(df)

        assert "filename" not in result.columns
