import pytest
from unittest.mock import MagicMock
from sqlalchemy.orm import Session

from federated_pneumonia_detection.src.control.analytics.internals.extractors.centralized import (
    CentralizedMetricExtractor,
)
from federated_pneumonia_detection.src.control.analytics.internals.extractors.federated import (
    FederatedMetricExtractor,
)


class TestCentralizedMetricExtractor:
    @pytest.fixture
    def mock_crud(self):
        return MagicMock()

    @pytest.fixture
    def extractor(self, mock_crud):
        return CentralizedMetricExtractor(crud=mock_crud)

    @pytest.fixture
    def mock_db(self):
        return MagicMock(spec=Session)

    def test_get_best_metric_mapped(self, extractor, mock_crud, mock_db):
        # Setup
        mock_best = MagicMock()
        mock_best.metric_value = 0.95
        mock_crud.get_best_metric.return_value = mock_best

        # Execute
        result = extractor.get_best_metric(mock_db, 1, "accuracy")

        # Assert
        assert result == 0.95
        mock_crud.get_best_metric.assert_called_once_with(
            mock_db, 1, "val_acc", maximize=True
        )

    def test_get_best_metric_unmapped(self, extractor, mock_crud, mock_db):
        # Setup
        mock_best = MagicMock()
        mock_best.metric_value = 0.8
        mock_crud.get_best_metric.return_value = mock_best

        # Execute
        result = extractor.get_best_metric(mock_db, 1, "custom_metric")

        # Assert
        assert result == 0.8
        mock_crud.get_best_metric.assert_called_once_with(
            mock_db, 1, "val_custom_metric", maximize=True
        )

    def test_get_best_metric_none(self, extractor, mock_crud, mock_db):
        # Setup
        mock_crud.get_best_metric.return_value = None

        # Execute
        result = extractor.get_best_metric(mock_db, 1, "accuracy")

        # Assert
        assert result is None


class TestFederatedMetricExtractor:
    @pytest.fixture
    def mock_crud(self):
        return MagicMock()

    @pytest.fixture
    def extractor(self, mock_crud):
        return FederatedMetricExtractor(crud=mock_crud)

    @pytest.fixture
    def mock_db(self):
        return MagicMock(spec=Session)

    def test_get_best_metric_success(self, extractor, mock_crud, mock_db):
        # Setup
        mock_crud.get_summary_stats.return_value = {
            "best_accuracy": {"value": 0.92, "round": 5}
        }

        # Execute
        result = extractor.get_best_metric(mock_db, 1, "accuracy")

        # Assert
        assert result == 0.92
        mock_crud.get_summary_stats.assert_called_once_with(mock_db, 1)

    def test_get_best_metric_no_summary(self, extractor, mock_crud, mock_db):
        # Setup
        mock_crud.get_summary_stats.return_value = None

        # Execute
        result = extractor.get_best_metric(mock_db, 1, "accuracy")

        # Assert
        assert result is None

    def test_get_best_metric_missing_key(self, extractor, mock_crud, mock_db):
        # Setup
        mock_crud.get_summary_stats.return_value = {"best_precision": {"value": 0.9}}

        # Execute
        result = extractor.get_best_metric(mock_db, 1, "accuracy")

        # Assert
        assert result is None
