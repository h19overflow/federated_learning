from unittest.mock import MagicMock

from federated_pneumonia_detection.src.control.analytics.facade import AnalyticsFacade


def test_analytics_facade_initialization_full():
    """Test AnalyticsFacade initialization with all services provided."""
    mock_metrics = MagicMock()
    mock_summary = MagicMock()
    mock_ranking = MagicMock()
    mock_export = MagicMock()
    mock_backfill = MagicMock()

    facade = AnalyticsFacade(
        metrics=mock_metrics,
        summary=mock_summary,
        ranking=mock_ranking,
        export=mock_export,
        backfill=mock_backfill,
    )

    assert facade.metrics == mock_metrics
    assert facade.summary == mock_summary
    assert facade.ranking == mock_ranking
    assert facade.export == mock_export
    assert facade.backfill == mock_backfill


def test_analytics_facade_initialization_empty():
    """Test AnalyticsFacade initialization with no services."""
    facade = AnalyticsFacade()

    assert facade.metrics is None
    assert facade.summary is None
    assert facade.ranking is None
    assert facade.export is None
    assert facade.backfill is None


def test_analytics_facade_initialization_partial():
    """Test AnalyticsFacade initialization with partial services."""
    mock_metrics = MagicMock()

    facade = AnalyticsFacade(metrics=mock_metrics)

    assert facade.metrics == mock_metrics
    assert facade.summary is None
    assert facade.ranking is None
    assert facade.export is None
    assert facade.backfill is None
