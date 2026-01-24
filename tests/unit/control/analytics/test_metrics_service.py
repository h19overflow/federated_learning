"""Unit tests for MetricsService.

These tests verify the MetricsService functionality including:
- Metric extraction strategies
- Run aggregation
- Transformation utilities
- Caching behavior
"""

import os
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.orm import Session

# Set environment variables before any imports
os.environ.update(
    {
        "POSTGRES_DB": "test_db",
        "POSTGRES_USER": "test_user",
        "POSTGRES_PASSWORD": "test_password",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB_URI": "postgresql://test_user:test_password@localhost:5432/test_db",
        "GEMINI_API_KEY": "test_key",
        "GOOGLE_API_KEY": "test_key",
        "BASE_LLM": "gemini-1.5-flash",
        "LANGSMITH_TRACING": "false",
        "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
        "LANGSMITH_API_KEY": "test_key",
        "LANGSMITH_PROJECT": "test_project",
    }
)

from federated_pneumonia_detection.src.control.analytics.cache import CacheProvider
from federated_pneumonia_detection.src.control.analytics.metrics_service import (
    MetricsService,
    CentralizedMetricExtractor,
    FederatedMetricExtractor,
    _calculate_summary_statistics,
    _find_best_epoch,
    _transform_run_to_results,
)


class TestMetricExtractor:
    """Test metric extractor strategies."""

    def test_federated_extractor_best_metric(self, db_session: Session):
        """Test federated metric extraction."""
        mock_crud = Mock()
        mock_crud.get_summary_stats.return_value = {
            "best_accuracy": {"value": 0.90, "round": 2},
            "best_precision": {"value": 0.88, "round": 2},
            "best_recall": {"value": 0.92, "round": 2},
            "best_f1_score": {"value": 0.90, "round": 2},
        }

        extractor = FederatedMetricExtractor(mock_crud)
        accuracy = extractor.get_best_metric(db_session, 1, "accuracy")

        assert accuracy == 0.90
        mock_crud.get_summary_stats.assert_called_once_with(db_session, 1)

    def test_federated_extractor_no_data(self, db_session: Session):
        """Test federated extractor with no data."""
        mock_crud = Mock()
        mock_crud.get_summary_stats.return_value = {}

        extractor = FederatedMetricExtractor(mock_crud)
        accuracy = extractor.get_best_metric(db_session, 1, "accuracy")

        assert accuracy is None

    def test_centralized_extractor_best_metric(self, db_session: Session):
        """Test centralized metric extraction."""
        mock_crud = Mock()
        mock_metric = Mock(metric_value=0.92)
        mock_crud.get_best_metric.return_value = mock_metric

        extractor = CentralizedMetricExtractor(mock_crud)
        accuracy = extractor.get_best_metric(db_session, 1, "accuracy")

        assert accuracy == 0.92
        mock_crud.get_best_metric.assert_called_once_with(
            db_session, 1, "val_acc", maximize=True
        )

    def test_centralized_extractor_metric_name_mapping(self, db_session: Session):
        """Test centralized extractor metric name mapping."""
        mock_crud = Mock()
        mock_metric = Mock(metric_value=0.88)
        mock_crud.get_best_metric.return_value = mock_metric

        extractor = CentralizedMetricExtractor(mock_crud)
        precision = extractor.get_best_metric(db_session, 1, "precision")

        assert precision == 0.88
        # Should use mapped name "val_precision"
        mock_crud.get_best_metric.assert_called_once_with(
            db_session, 1, "val_precision", maximize=True
        )

    def test_centralized_extractor_no_data(self, db_session: Session):
        """Test centralized extractor with no data."""
        mock_crud = Mock()
        mock_crud.get_best_metric.return_value = None

        extractor = CentralizedMetricExtractor(mock_crud)
        accuracy = extractor.get_best_metric(db_session, 1, "accuracy")

        assert accuracy is None


class TestRunAggregator:
    """Test run statistics aggregation."""

    def test_calculate_statistics_with_runs(
        self, mock_cache: CacheProvider, db_session: Session
    ):
        """Test aggregation with multiple runs."""
        service = MetricsService(cache=mock_cache)

        # Mock extractor to return consistent metrics
        mock_extractor = Mock()
        mock_extractor.get_best_metric.side_effect = lambda db, rid, name: {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.92,
            "f1_score": 0.90,
        }.get(name)

        # Create mock runs
        run1 = Mock(
            id=1,
            training_mode="centralized",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1),
        )
        run2 = Mock(
            id=2,
            training_mode="federated",
            start_time=datetime.now() - timedelta(hours=3),
            end_time=datetime.now() - timedelta(hours=2),
        )

        # Mock the extractor factory
        with patch.object(
            service, "_get_metric_extractor", return_value=mock_extractor
        ):
            stats = service._calculate_run_statistics(db_session, [run1, run2])

            assert stats["count"] == 2
            assert stats["avg_accuracy"] == 0.90
            assert stats["avg_precision"] == 0.88
            assert stats["avg_recall"] == 0.92
            assert stats["avg_f1"] == 0.90
            assert stats["avg_duration_minutes"] is not None

    def test_calculate_statistics_empty(
        self, mock_cache: CacheProvider, db_session: Session
    ):
        """Test aggregation with empty run list."""
        service = MetricsService(cache=mock_cache)

        stats = service._calculate_run_statistics(db_session, [])

        assert stats["count"] == 0
        assert stats["avg_accuracy"] is None
        assert stats["avg_precision"] is None
        assert stats["avg_recall"] is None
        assert stats["avg_f1"] is None
        assert stats["avg_duration_minutes"] is None

    def test_safe_average(self, mock_cache: CacheProvider):
        """Test safe average calculation."""
        service = MetricsService(cache=mock_cache)

        assert service._safe_average([1.0, 2.0, 3.0]) == 2.0
        assert service._safe_average([]) is None
        assert service._safe_average([0.85, 0.90]) == pytest.approx(0.875, rel=1e-3)


class TestRunTransformation:
    """Test run transformation utilities."""

    def test_calculate_summary_statistics(self):
        """Test confusion matrix statistics calculation."""
        cm = {
            "true_positives": 90,
            "true_negatives": 80,
            "false_positives": 10,
            "false_negatives": 20,
        }

        stats = _calculate_summary_statistics(cm)

        assert stats["sensitivity"] == pytest.approx(0.8182, rel=1e-3)
        assert stats["specificity"] == pytest.approx(0.8889, rel=1e-3)
        assert stats["precision_cm"] == pytest.approx(0.9000, rel=1e-3)
        assert stats["accuracy_cm"] == pytest.approx(0.8500, rel=1e-3)
        assert stats["f1_cm"] == pytest.approx(0.8571, rel=1e-3)

    def test_calculate_summary_statistics_negative_values(self):
        """Test confusion matrix with negative values raises error."""
        cm = {
            "true_positives": -10,
            "true_negatives": 80,
            "false_positives": 10,
            "false_negatives": 20,
        }

        with pytest.raises(ValueError):
            _calculate_summary_statistics(cm)

    def test_calculate_summary_statistics_zero_total(self):
        """Test confusion matrix with zero total samples."""
        cm = {
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

        stats = _calculate_summary_statistics(cm)

        assert stats["sensitivity"] == 0.0
        assert stats["specificity"] == 0.0
        assert stats["precision_cm"] == 0.0
        assert stats["accuracy_cm"] == 0.0
        assert stats["f1_cm"] == 0.0

    def test_find_best_epoch(self):
        """Test best epoch finding logic."""
        training_history = [
            {"epoch": 1, "val_acc": 0.80},
            {"epoch": 2, "val_acc": 0.85},
            {"epoch": 3, "val_acc": 0.90},
            {"epoch": 4, "val_acc": 0.88},
        ]

        best_epoch = _find_best_epoch(training_history)

        assert best_epoch == 3

    def test_find_best_epoch_empty_history(self):
        """Test best epoch finding with empty history."""
        best_epoch = _find_best_epoch([])

        assert best_epoch == 1

    def test_transform_centralized_run(self):
        """Test centralized run transformation."""
        run = Mock(
            id=1,
            training_mode="centralized",
            status="completed",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1),
        )
        run.metrics = []
        run.server_evaluations = []

        results = _transform_run_to_results(run)

        assert results["experiment_id"] == "run_1"
        assert results["status"] == "completed"
        assert results["total_epochs"] == 0
        assert results["training_history"] == []

    def test_transform_federated_run(self):
        """Test federated run transformation."""
        run = Mock(
            id=2,
            training_mode="federated",
            status="completed",
            start_time=datetime.now() - timedelta(hours=3),
            end_time=datetime.now() - timedelta(hours=2),
        )
        run.metrics = []

        # Add mock server evaluation
        mock_eval = Mock(
            round_number=1,
            loss=0.4,
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            auroc=0.87,
            true_positives=85,
            true_negatives=80,
            false_positives=10,
            false_negatives=15,
        )
        run.server_evaluations = [mock_eval]

        results = _transform_run_to_results(run)

        assert results["experiment_id"] == "run_2"
        assert results["status"] == "completed"
        assert results["total_epochs"] == 1
        assert len(results["training_history"]) == 1
        assert results["training_history"][0]["epoch"] == 1
        assert results["training_history"][0]["val_acc"] == 0.85

    def test_transform_with_persisted_stats(self):
        """Test run transformation with persisted stats."""
        run = Mock(
            id=1,
            training_mode="centralized",
            status="completed",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1),
        )
        run.metrics = []
        run.server_evaluations = []

        persisted_stats = {
            "sensitivity": 0.90,
            "specificity": 0.85,
            "precision_cm": 0.88,
            "accuracy_cm": 0.87,
            "f1_cm": 0.89,
        }

        results = _transform_run_to_results(run, persisted_stats=persisted_stats)

        assert results["confusion_matrix"] is not None
        assert results["confusion_matrix"]["sensitivity"] == 0.90
        assert results["confusion_matrix"]["f1_cm"] == 0.89


class TestMetricsService:
    """Test MetricsService methods."""

    def test_get_run_metrics_caches_results(
        self, mock_cache: CacheProvider, db_session: Session
    ):
        """Test that get_run_metrics caches results correctly."""
        # Mock CRUD to return a run
        mock_run = Mock(
            id=1,
            training_mode="centralized",
            status="completed",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1),
            metrics=[],
            server_evaluations=[],
        )

        # Create mock CRUD objects
        mock_run_crud = Mock()
        mock_run_crud.get_with_metrics.return_value = mock_run
        mock_run_metric_crud = Mock()
        mock_server_evaluation_crud = Mock()

        service = MetricsService(
            cache=mock_cache,
            run_crud_obj=mock_run_crud,
            run_metric_crud_obj=mock_run_metric_crud,
            server_evaluation_crud_obj=mock_server_evaluation_crud,
        )

        # First call - should compute
        result1 = service.get_run_metrics(db_session, 1)

        # Verify CRUD was called
        mock_run_crud.get_with_metrics.assert_called_once_with(db_session, 1)

        # Reset mock
        mock_run_crud.get_with_metrics.reset_mock()

        # Second call - should use cache (CRUD not called)
        result2 = service.get_run_metrics(db_session, 1)

        # Verify CRUD was NOT called second time (cached)
        mock_run_crud.get_with_metrics.assert_not_called()

        # Results should be identical
        assert result1 == result2

    def test_get_run_metrics_not_found(
        self, mock_cache: CacheProvider, db_session: Session
    ):
        """Test get_run_metrics with non-existent run."""
        mock_run_crud = Mock()
        mock_run_crud.get_with_metrics.return_value = None
        mock_run_metric_crud = Mock()
        mock_server_evaluation_crud = Mock()

        service = MetricsService(
            cache=mock_cache,
            run_crud_obj=mock_run_crud,
            run_metric_crud_obj=mock_run_metric_crud,
            server_evaluation_crud_obj=mock_server_evaluation_crud,
        )

        with pytest.raises(ValueError, match="Run 1 not found"):
            service.get_run_metrics(db_session, 1)

    def test_get_run_detail(self, mock_cache: CacheProvider, db_session: Session):
        """Test get_run_detail."""
        # Mock run with metrics
        mock_run = Mock(
            id=1,
            training_mode="centralized",
            status="completed",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1),
            metrics=[],
            server_evaluations=[],
        )

        mock_run_crud = Mock()
        mock_run_crud.get_with_metrics.return_value = mock_run
        mock_run_metric_crud = Mock()
        mock_server_evaluation_crud = Mock()

        service = MetricsService(
            cache=mock_cache,
            run_crud_obj=mock_run_crud,
            run_metric_crud_obj=mock_run_metric_crud,
            server_evaluation_crud_obj=mock_server_evaluation_crud,
        )

        # Mock extractor to return metrics
        mock_extractor = Mock()
        mock_extractor.get_best_metric.return_value = 0.90

        with patch.object(
            service, "_get_metric_extractor", return_value=mock_extractor
        ):
            detail = service.get_run_detail(db_session, 1)

            assert detail["run_id"] == 1
            assert detail["training_mode"] == "centralized"
            assert detail["best_accuracy"] == 0.90
            assert detail["status"] == "completed"

    def test_get_run_detail_no_metrics(
        self, mock_cache: CacheProvider, db_session: Session
    ):
        """Test get_run_detail with no metrics."""
        mock_run = Mock(
            id=1,
            training_mode="centralized",
            status="completed",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1),
            metrics=[],
            server_evaluations=[],
        )

        mock_run_crud = Mock()
        mock_run_crud.get_with_metrics.return_value = mock_run
        mock_run_metric_crud = Mock()
        mock_server_evaluation_crud = Mock()

        service = MetricsService(
            cache=mock_cache,
            run_crud_obj=mock_run_crud,
            run_metric_crud_obj=mock_run_metric_crud,
            server_evaluation_crud_obj=mock_server_evaluation_crud,
        )

        # Mock extractor to return None
        mock_extractor = Mock()
        mock_extractor.get_best_metric.return_value = None

        with patch.object(
            service, "_get_metric_extractor", return_value=mock_extractor
        ):
            with pytest.raises(ValueError, match="No metrics found for run 1"):
                service.get_run_detail(db_session, 1)

    def test_get_analytics_summary_empty(
        self, mock_cache: CacheProvider, db_session: Session
    ):
        """Test analytics summary when no runs found."""
        mock_run_crud = Mock()
        mock_run_crud.get_by_status_and_mode.return_value = []
        mock_run_crud.get_by_status.return_value = []
        mock_run_metric_crud = Mock()
        mock_server_evaluation_crud = Mock()

        service = MetricsService(
            cache=mock_cache,
            run_crud_obj=mock_run_crud,
            run_metric_crud_obj=mock_run_metric_crud,
            server_evaluation_crud_obj=mock_server_evaluation_crud,
        )

        summary = service.get_analytics_summary(db_session, filters={"status": "test"})

        assert summary["total_runs"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["centralized"]["count"] == 0
        assert summary["federated"]["count"] == 0
        assert summary["top_runs"] == []

    def test_get_analytics_summary_with_runs(
        self, mock_cache: CacheProvider, db_session: Session
    ):
        """Test analytics summary with runs."""
        mock_run = Mock(
            id=1,
            training_mode="centralized",
            status="completed",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1),
        )

        mock_run_crud = Mock()
        mock_run_crud.get_by_status_and_mode.return_value = [mock_run]
        mock_run_crud.get_by_status.return_value = [mock_run]
        mock_run_metric_crud = Mock()
        mock_server_evaluation_crud = Mock()

        service = MetricsService(
            cache=mock_cache,
            run_crud_obj=mock_run_crud,
            run_metric_crud_obj=mock_run_metric_crud,
            server_evaluation_crud_obj=mock_server_evaluation_crud,
        )

        # Mock extractor to return metrics
        mock_extractor = Mock()
        mock_extractor.get_best_metric.return_value = 0.90

        with patch.object(
            service, "_get_metric_extractor", return_value=mock_extractor
        ):
            summary = service.get_analytics_summary(
                db_session, filters={"status": "completed"}
            )

            assert summary["total_runs"] == 1
            assert summary["success_rate"] == 1.0
            assert summary["centralized"]["count"] == 1

    def test_get_analytics_summary_with_days_filter(
        self, mock_cache: CacheProvider, db_session: Session
    ):
        """Test analytics summary with time filter."""
        # Old run (10 days ago)
        old_run = Mock(
            id=1,
            training_mode="centralized",
            status="completed",
            start_time=datetime.now() - timedelta(days=10),
            end_time=datetime.now() - timedelta(days=10, hours=1),
        )
        # Recent run (1 day ago)
        recent_run = Mock(
            id=2,
            training_mode="centralized",
            status="completed",
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now() - timedelta(days=1, hours=1),
        )

        mock_run_crud = Mock()
        mock_run_crud.get_by_status_and_mode.return_value = [old_run, recent_run]
        mock_run_crud.get_by_status.return_value = [old_run, recent_run]
        mock_run_metric_crud = Mock()
        mock_server_evaluation_crud = Mock()

        service = MetricsService(
            cache=mock_cache,
            run_crud_obj=mock_run_crud,
            run_metric_crud_obj=mock_run_metric_crud,
            server_evaluation_crud_obj=mock_server_evaluation_crud,
        )

        summary = service.get_analytics_summary(
            db_session, filters={"status": "completed", "days": 7}
        )

        # Only recent run should be included
        assert summary["total_runs"] == 1

        service = MetricsService(cache=mock_cache)
        service._run_crud.get_by_status_and_mode.return_value = [old_run, recent_run]
        service._run_crud.get_by_status.return_value = [old_run, recent_run]

        summary = service.get_analytics_summary(
            db_session, filters={"status": "completed", "days": 7}
        )

        # Only recent run should be included
        assert summary["total_runs"] == 1

    def test_get_top_runs(self, mock_cache: CacheProvider):
        """Test top runs ranking."""
        service = MetricsService(cache=mock_cache)

        runs = [
            {"run_id": 1, "best_accuracy": 0.85},
            {"run_id": 2, "best_accuracy": 0.92},
            {"run_id": 3, "best_accuracy": 0.88},
        ]

        top_runs = service._get_top_runs(runs, metric="best_accuracy", limit=2)

        assert len(top_runs) == 2
        assert top_runs[0]["run_id"] == 2
        assert top_runs[1]["run_id"] == 3

    def test_extract_run_detail(self, mock_cache: CacheProvider, db_session: Session):
        """Test run detail extraction."""
        mock_run = Mock(
            id=1,
            training_mode="centralized",
            status="completed",
            start_time=datetime.now() - timedelta(hours=2),
            end_time=datetime.now() - timedelta(hours=1),
        )

        service = MetricsService(cache=mock_cache)

        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.get_best_metric.return_value = 0.90

        with patch.object(
            service, "_get_metric_extractor", return_value=mock_extractor
        ):
            detail = service._extract_run_detail(db_session, mock_run)

            assert detail["run_id"] == 1
            assert detail["training_mode"] == "centralized"
            assert detail["best_accuracy"] == 0.90
            assert detail["duration_minutes"] == pytest.approx(60.0, rel=1e-1)
            assert detail["status"] == "completed"

    def test_get_metric_extractor_factory(self, mock_cache: CacheProvider):
        """Test metric extractor factory method."""
        centralized_run = Mock(training_mode="centralized")
        federated_run = Mock(training_mode="federated")

        service = MetricsService(cache=mock_cache)

        centralized_extractor = service._get_metric_extractor(centralized_run)
        federated_extractor = service._get_metric_extractor(federated_run)

        assert isinstance(centralized_extractor, CentralizedMetricExtractor)
        assert isinstance(federated_extractor, FederatedMetricExtractor)


# Pytest fixtures


@pytest.fixture
def mock_cache() -> CacheProvider:
    """Create a mock cache provider."""
    return CacheProvider(ttl=600, maxsize=100)


@pytest.fixture
def db_session() -> Session:
    """Create a mock database session."""
    return Mock(spec=Session)
