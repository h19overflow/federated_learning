import pytest
from unittest.mock import MagicMock, patch
from federated_pneumonia_detection.src.control.analytics.internals.services.summary_service import (
    SummaryService,
)
from federated_pneumonia_detection.src.boundary.models import (
    RunMetric,
    ServerEvaluation,
)


@pytest.fixture
def summary_service(
    mock_cache, mock_run_crud, mock_run_metric_crud, mock_server_eval_crud
):
    """Fixture for SummaryService with mocked dependencies."""
    return SummaryService(
        cache=mock_cache,
        run_crud_obj=mock_run_crud,
        run_metric_crud_obj=mock_run_metric_crud,
        server_evaluation_crud_obj=mock_server_eval_crud,
    )


def test_get_run_summary_success(
    summary_service, mock_session, mock_run_crud, sample_run
):
    """Test successful retrieval of run summary."""
    # Setup
    sample_run.metrics = []
    sample_run.run_description = "Test run"
    mock_run_crud.get_with_metrics.return_value = sample_run

    # Execute
    summary = summary_service.get_run_summary(mock_session, 1)

    # Assert
    assert summary["id"] == 1
    assert summary["training_mode"] == "centralized"
    assert "best_val_recall" in summary
    assert "best_val_accuracy" in summary
    assert "federated_info" in summary
    assert "final_epoch_stats" in summary
    mock_run_crud.get_with_metrics.assert_called_once_with(mock_session, 1)


def test_get_run_summary_not_found(summary_service, mock_session, mock_run_crud):
    """Test error when run is not found."""
    # Setup
    mock_run_crud.get_with_metrics.return_value = None

    # Execute & Assert
    with pytest.raises(ValueError, match="Run 999 not found"):
        summary_service.get_run_summary(mock_session, 999)


def test_list_runs_with_summaries_filters(summary_service, mock_session, mock_run_crud):
    """Verify filters are passed correctly to CRUD."""
    # Setup
    mock_run_crud.list_with_filters.return_value = ([], 0)

    # Execute
    summary_service.list_runs_with_summaries(
        mock_session,
        status="completed",
        training_mode="centralized",
        limit=5,
        offset=10,
    )

    # Assert
    mock_run_crud.list_with_filters.assert_called_once_with(
        mock_session,
        status="completed",
        training_mode="centralized",
        sort_by="start_time",
        sort_order="desc",
        limit=5,
        offset=10,
    )


def test_list_runs_with_summaries_graceful_failure(
    summary_service, mock_session, mock_run_crud, sample_run
):
    """Test graceful failure when summary building fails for one run."""
    # Setup
    mock_run_crud.list_with_filters.return_value = ([sample_run], 1)

    # Patch the internal _build_run_summary to raise an exception
    with patch.object(
        summary_service, "_build_run_summary", side_effect=Exception("Build failed")
    ):
        # Execute
        result = summary_service.list_runs_with_summaries(mock_session)

        # Assert
        assert len(result["runs"]) == 1
        assert result["runs"][0]["id"] == sample_run.id
        assert result["runs"][0]["error"] == "Summary unavailable"
        assert result["total"] == 1


def test_build_federated_info(
    summary_service, mock_session, mock_server_eval_crud, sample_federated_run
):
    """Test building federated info from server evaluations."""
    # Setup
    eval1 = MagicMock()
    eval1.round_number = 1
    eval1.accuracy = 0.8
    eval1.recall = 0.7

    eval2 = MagicMock()
    eval2.round_number = 2
    eval2.accuracy = 0.9
    eval2.recall = 0.85

    mock_server_eval_crud.get_by_run.return_value = [eval1, eval2]
    sample_federated_run.clients = [MagicMock(), MagicMock()]  # 2 clients

    # Execute
    info = summary_service._build_federated_info(sample_federated_run, mock_session)

    # Assert
    assert info["num_rounds"] == 2
    assert info["num_clients"] == 2
    assert info["best_accuracy"] == 0.9
    assert info["best_recall"] == 0.85
    assert info["latest_round"] == 2
    assert info["latest_accuracy"] == 0.9


def test_get_final_epoch_stats_centralized(summary_service, mock_session, sample_run):
    """Test extraction of final epoch stats for centralized run."""
    # Setup
    m1 = MagicMock(spec=RunMetric)
    m1.metric_name = "final_sensitivity"
    m1.metric_value = 0.95
    m2 = MagicMock(spec=RunMetric)
    m2.metric_name = "final_specificity"
    m2.metric_value = 0.90
    m3 = MagicMock(spec=RunMetric)
    m3.metric_name = "final_precision_cm"
    m3.metric_value = 0.92
    m4 = MagicMock(spec=RunMetric)
    m4.metric_name = "final_accuracy_cm"
    m4.metric_value = 0.93
    m5 = MagicMock(spec=RunMetric)
    m5.metric_name = "final_f1_cm"
    m5.metric_value = 0.94

    # Mock the query chain: db.query(RunMetric).filter(...).all()
    mock_session.query.return_value.filter.return_value.all.return_value = [
        m1,
        m2,
        m3,
        m4,
        m5,
    ]

    # Execute
    stats = summary_service._get_final_epoch_stats(sample_run, mock_session)

    # Assert
    assert stats == {
        "sensitivity": 0.95,
        "specificity": 0.90,
        "precision_cm": 0.92,
        "accuracy_cm": 0.93,
        "f1_cm": 0.94,
    }


def test_get_final_epoch_stats_federated(
    summary_service, mock_session, sample_federated_run
):
    """Test extraction of final epoch stats for federated run."""
    # Setup
    last_eval = MagicMock(spec=ServerEvaluation)
    last_eval.additional_metrics = {
        "final_epoch_stats": {"sensitivity": 0.88, "specificity": 0.85}
    }

    # Mock the query chain: db.query(ServerEvaluation).filter(...).order_by(...).first()
    mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = last_eval

    # Execute
    stats = summary_service._get_final_epoch_stats(sample_federated_run, mock_session)

    # Assert
    assert stats == {"sensitivity": 0.88, "specificity": 0.85}
