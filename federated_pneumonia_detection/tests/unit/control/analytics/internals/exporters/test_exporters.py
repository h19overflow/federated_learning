import json
import csv
from io import StringIO
import pytest
from federated_pneumonia_detection.src.control.analytics.internals.exporters.csv_exporter import (
    CSVExporter,
)
from federated_pneumonia_detection.src.control.analytics.internals.exporters.json_exporter import (
    JSONExporter,
)
from federated_pneumonia_detection.src.control.analytics.internals.exporters.text_exporter import (
    TextReportExporter,
)


class TestCSVExporter:
    @pytest.fixture
    def exporter(self):
        return CSVExporter()

    def test_export_success(self, exporter):
        # Setup
        data = {
            "training_history": [{"epoch": 1, "acc": 0.8}, {"epoch": 2, "acc": 0.9}]
        }

        # Execute
        result = exporter.export(data)

        # Assert
        assert isinstance(result, str)
        f = StringIO(result)
        reader = csv.DictReader(f)
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["epoch"] == "1"
        assert rows[0]["acc"] == "0.8"
        assert rows[1]["epoch"] == "2"
        assert rows[1]["acc"] == "0.9"
        assert set(reader.fieldnames) == {"epoch", "acc"}

    def test_export_empty(self, exporter):
        # Setup
        data = {"training_history": []}

        # Execute
        result = exporter.export(data)

        # Assert
        assert result == ""

    def test_export_missing_key(self, exporter):
        # Setup
        data = {}

        # Execute
        result = exporter.export(data)

        # Assert
        assert result == ""

    def test_metadata(self, exporter):
        assert exporter.get_media_type() == "text/csv"
        assert exporter.get_file_extension() == "csv"


class TestJSONExporter:
    @pytest.fixture
    def exporter(self):
        return JSONExporter()

    def test_export_success(self, exporter):
        # Setup
        data = {"summary": {"best_acc": 0.95}, "history": [1, 2, 3]}

        # Execute
        result = exporter.export(data)

        # Assert
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == data

    def test_metadata(self, exporter):
        assert exporter.get_media_type() == "application/json"
        assert exporter.get_file_extension() == "json"


class TestTextReportExporter:
    @pytest.fixture
    def exporter(self):
        return TextReportExporter()

    def test_export_success(self, exporter):
        # Setup
        data = {
            "metadata": {"experiment_name": "Test Exp", "total_epochs": 10},
            "status": "completed",
            "final_metrics": {"accuracy": 0.95, "recall": 0.94},
            "training_history": [
                {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_acc": 0.8}
            ],
        }

        # Execute
        result = exporter.export(data)

        # Assert
        assert "TRAINING RUN SUMMARY REPORT" in result
        assert "Experiment: Test Exp" in result
        assert "Status: completed" in result
        assert "Acc      0.9500" in result
        assert "E1  | TL:0.500 VL:0.400 VA:80.0%" in result

    def test_metadata(self, exporter):
        assert exporter.get_media_type() == "text/plain"
        assert exporter.get_file_extension() == "txt"
