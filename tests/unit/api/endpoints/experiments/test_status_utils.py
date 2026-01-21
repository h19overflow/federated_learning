"""
Unit tests for status utilities in experiment endpoints.

Tests cover:
- Finding experiment log files
- Calculating training progress
- Handling missing or incomplete logs
- Edge cases for progress calculation
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from federated_pneumonia_detection.src.api.endpoints.experiments.utils.status_utils import (
    LOGS_DIR,
    calculate_progress,
    find_experiment_log_file,
)


class TestFindExperimentLogFile:
    """Test find_experiment_log_file function."""

    def test_find_experiment_log_file_exists(self, sample_experiment_log):
        """Test finding existing experiment log file."""
        # Create logs directory
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Create log file
        log_file = LOGS_DIR / "test_experiment.json"
        with open(log_file, "w") as f:
            json.dump(sample_experiment_log, f)

        # Find the log
        result = find_experiment_log_file("test_experiment")

        assert result is not None
        assert result == log_file

    def test_find_experiment_log_file_not_found(self, tmp_path):
        """Test finding non-existent experiment log."""
        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.status_utils.LOGS_DIR",
            tmp_path,
        ):
            result = find_experiment_log_file("nonexistent")
            assert result is None

    def test_find_experiment_log_partial_match(self, tmp_path):
        """Test finding log with partial experiment ID match."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        # Create log with long name
        log_data = {"metadata": {"status": "completed"}}
        log_file = logs_dir / "experiment_2025_01_21_12345.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f)

        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.status_utils.LOGS_DIR",
            logs_dir,
        ):
            result = find_experiment_log_file("2025_01_21")
            assert result is not None
            assert "2025_01_21" in str(result)

    def test_find_experiment_log_logs_dir_not_exists(self):
        """Test behavior when logs directory doesn't exist."""
        with patch(
            "federated_pneumonia_detection.src.api.endpoints.experiments.utils.status_utils.LOGS_DIR",
            Path("/nonexistent/path"),
        ):
            result = find_experiment_log_file("test")
            assert result is None


class TestCalculateProgress:
    """Test calculate_progress function."""

    def test_calculate_progress_no_epochs(self):
        """Test progress calculation with no epochs."""
        log_data = {"epochs": []}
        progress = calculate_progress(log_data)
        assert progress == 0.0

    def test_calculate_progress_partial_training(self, sample_experiment_log):
        """Test progress calculation during training (3/10 epochs)."""
        progress = calculate_progress(sample_experiment_log)
        assert progress == pytest.approx(30.0, rel=0.1)

    def test_calculate_progress_completed_training(
        self,
        sample_completed_experiment_log,
    ):
        """Test progress calculation for completed training."""
        progress = calculate_progress(sample_completed_experiment_log)
        assert progress == 100.0

    def test_calculate_progress_custom_total_epochs(self):
        """Test progress calculation with custom total_epochs."""
        log_data = {
            "epochs": [
                {"type": "epoch_start", "total_epochs": 5},
                {"type": "epoch_end"},
                {"type": "epoch_start", "total_epochs": 5},
                {"type": "epoch_end"},
            ],
        }
        progress = calculate_progress(log_data)
        assert progress == pytest.approx(40.0, rel=0.1)

    def test_calculate_progress_total_epochs_in_metadata(self):
        """Test progress calculation with total_epochs from first epoch_start."""
        log_data = {
            "epochs": [
                {"type": "epoch_start", "epoch": 1, "total_epochs": 20},
                {"type": "epoch_end", "epoch": 1},
                {"type": "epoch_start", "epoch": 2, "total_epochs": 20},
                {"type": "epoch_end", "epoch": 2},
                {"type": "epoch_start", "epoch": 3, "total_epochs": 20},
            ],
        }
        progress = calculate_progress(log_data)
        assert progress == pytest.approx(10.0, rel=0.1)

    def test_calculate_progress_exceeds_100(self):
        """Test progress is capped at 100% even if data suggests more."""
        log_data = {
            "epochs": [
                {"type": "epoch_start", "total_epochs": 5},
                {"type": "epoch_end"},
            ]
            * 10,  # More completed epochs than total
        }
        progress = calculate_progress(log_data)
        assert progress == 100.0

    def test_calculate_progress_zero_total_epochs(self):
        """Test progress with zero total_epochs (edge case)."""
        log_data = {
            "epochs": [
                {"type": "epoch_start", "total_epochs": 0},
                {"type": "epoch_end"},
            ],
        }
        progress = calculate_progress(log_data)
        assert progress == 0.0

    def test_calculate_progress_negative_total_epochs(self):
        """Test progress with negative total_epochs (edge case)."""
        log_data = {
            "epochs": [
                {"type": "epoch_start", "total_epochs": -1},
                {"type": "epoch_end"},
            ],
        }
        progress = calculate_progress(log_data)
        assert progress == 0.0

    def test_calculate_progress_only_epoch_start_events(self):
        """Test progress with only epoch_start events (no epoch_end)."""
        log_data = {
            "epochs": [
                {"type": "epoch_start", "total_epochs": 10},
                {"type": "epoch_start", "epoch": 2, "total_epochs": 10},
                {"type": "epoch_start", "epoch": 3, "total_epochs": 10},
            ],
        }
        progress = calculate_progress(log_data)
        # Only epoch_end events count as completed
        assert progress == 0.0

    def test_calculate_progress_mixed_events(self):
        """Test progress with mixed event types."""
        log_data = {
            "epochs": [
                {"type": "epoch_start", "total_epochs": 10},
                {"type": "epoch_end"},
                {"type": "checkpoint"},
                {"type": "validation"},
                {"type": "epoch_start", "epoch": 2, "total_epochs": 10},
                {"type": "epoch_end", "epoch": 2},
                {"type": "epoch_start", "epoch": 3, "total_epochs": 10},
            ],
        }
        progress = calculate_progress(log_data)
        # 2 epoch_end events out of 10
        assert progress == pytest.approx(20.0, rel=0.1)

    def test_calculate_progress_default_total_epochs(self):
        """Test progress uses default (10) when total_epochs not found."""
        log_data = {
            "epochs": [
                {"type": "epoch_end"},
                {"type": "epoch_end"},
                {"type": "epoch_end"},
                {"type": "epoch_end"},
                {"type": "epoch_end"},
            ],
        }
        progress = calculate_progress(log_data)
        # 5 completed out of default 10
        assert progress == pytest.approx(50.0, rel=0.1)

    def test_calculate_progress_multiple_total_epochs(self):
        """Test progress uses first total_epochs found."""
        log_data = {
            "epochs": [
                {"type": "epoch_start", "total_epochs": 8},  # First one
                {"type": "epoch_end"},
                {"type": "epoch_start", "total_epochs": 10},  # Second one (ignored)
                {"type": "epoch_end"},
                {"type": "epoch_start", "total_epochs": 5},  # Third one (ignored)
            ],
        }
        progress = calculate_progress(log_data)
        # 2 completed out of first total (8)
        assert progress == pytest.approx(25.0, rel=0.1)
