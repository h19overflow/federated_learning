from unittest.mock import MagicMock, patch

from federated_pneumonia_detection.src.control.analytics.internals.services import (
    MetricsService,
)


class TestMetricsService:
    def test_get_run_metrics(self, mock_session, mock_cache, mock_run_crud, sample_run):
        """Test get_run_metrics returns correctly transformed data."""
        # Setup
        service = MetricsService(cache=mock_cache, run_crud_obj=mock_run_crud)
        mock_run_crud.get_with_metrics.return_value = sample_run
        sample_run.metrics = []

        # Execute
        result = service.get_run_metrics(mock_session, 1)

        # Assert
        assert result["experiment_id"] == "run_1"
        assert result["status"] == "completed"
        assert "final_metrics" in result
        assert "training_history" in result
        mock_run_crud.get_with_metrics.assert_called_once_with(mock_session, 1)

    def test_get_analytics_summary_no_runs(
        self, mock_session, mock_cache, mock_run_crud
    ):
        """Test summary when no runs are found."""
        # Setup
        service = MetricsService(cache=mock_cache, run_crud_obj=mock_run_crud)
        mock_run_crud.get_by_status_and_mode.return_value = []

        # Execute
        result = service.get_analytics_summary(mock_session, filters={})

        # Assert
        assert result["total_runs"] == 0
        assert result["success_rate"] == 0.0
        assert result["centralized"]["count"] == 0
        assert result["federated"]["count"] == 0
        assert result["top_runs"] == []

    def test_get_analytics_summary_mixed_runs(
        self, mock_session, mock_cache, mock_run_crud, sample_run, sample_federated_run
    ):
        """Test summary with both centralized and federated runs."""
        # Setup
        service = MetricsService(cache=mock_cache, run_crud_obj=mock_run_crud)
        mock_run_crud.get_by_status_and_mode.return_value = [
            sample_run,
            sample_federated_run,
        ]
        mock_run_crud.get_by_status.return_value = [sample_run, sample_federated_run]

        # Mock internal methods to avoid deep mocking extractors
        with patch.object(service, "_calculate_run_statistics") as mock_calc_stats:
            mock_calc_stats.return_value = {"count": 1, "avg_accuracy": 0.9}
            with patch.object(service, "_extract_run_detail") as mock_extract_detail:
                mock_extract_detail.return_value = {"run_id": 1, "best_accuracy": 0.9}

                # Execute
                result = service.get_analytics_summary(mock_session, filters={})

                # Assert
                assert result["total_runs"] == 2
                assert result["success_rate"] == 1.0
                assert result["centralized"] == {"count": 1, "avg_accuracy": 0.9}
                assert result["federated"] == {"count": 1, "avg_accuracy": 0.9}
                assert len(result["top_runs"]) == 2

    def test_calculate_run_statistics_math(self, mock_session, mock_cache, sample_run):
        """Verify math in _calculate_run_statistics (mean, precision handling)."""
        # Setup
        service = MetricsService(cache=mock_cache)

        # Create two runs with different metrics
        run1 = MagicMock()
        run1.id = 1
        run1.training_mode = "centralized"
        run1.start_time = MagicMock()
        run1.end_time = MagicMock()
        # 10 minutes
        (run1.end_time - run1.start_time).total_seconds.return_value = 600

        run2 = MagicMock()
        run2.id = 2
        run2.training_mode = "centralized"
        run2.start_time = MagicMock()
        run2.end_time = MagicMock()
        # 20 minutes
        (run2.end_time - run2.start_time).total_seconds.return_value = 1200

        with patch.object(service, "_get_metric_extractor") as mock_get_extractor:
            mock_extractor1 = MagicMock()
            mock_extractor1.get_best_metric.side_effect = [0.8, 0.7, 0.6, 0.5]

            mock_extractor2 = MagicMock()
            mock_extractor2.get_best_metric.side_effect = [0.9, 0.8, 0.7, 0.6]

            mock_get_extractor.side_effect = [mock_extractor1, mock_extractor2]

            # Execute
            stats = service._calculate_run_statistics(mock_session, [run1, run2])

            # Assert
            assert stats["count"] == 2
            # (0.8 + 0.9) / 2 = 0.85
            assert stats["avg_accuracy"] == 0.85
            # (0.7 + 0.8) / 2 = 0.75
            assert stats["avg_precision"] == 0.75
            # (0.6 + 0.7) / 2 = 0.65
            assert stats["avg_recall"] == 0.65
            # (0.5 + 0.6) / 2 = 0.55
            assert stats["avg_f1"] == 0.55
            # (10 + 20) / 2 = 15.0
            assert stats["avg_duration_minutes"] == 15.0

            # Verify rounding to 4 decimals
            mock_extractor1.get_best_metric.side_effect = [0.33333, 0, 0, 0]
            mock_extractor2.get_best_metric.side_effect = [0.66666, 0, 0, 0]
            mock_get_extractor.side_effect = [mock_extractor1, mock_extractor2]

            stats = service._calculate_run_statistics(mock_session, [run1, run2])
            # (0.33333 + 0.66666) / 2 = 0.499995 -> rounded to 0.5
            assert stats["avg_accuracy"] == 0.5
