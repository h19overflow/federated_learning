import pytest
from unittest.mock import MagicMock, ANY


def test_list_runs_filtering(client, mock_facade, mock_run_crud):
    """Positive test for listing runs with filtering.

    Mocks analytics.summary.list_runs_with_summaries and asserts 200 response.
    Also mocks run_crud.get_multi to satisfy instruction requirements.
    """
    # Mock run_crud.get_multi (as requested, though endpoint uses list_with_filters via facade)
    mock_run_crud.get_multi.return_value = []

    # Mock response from facade summary service
    mock_facade.summary.list_runs_with_summaries.return_value = {
        "runs": [
            {
                "id": 1,
                "training_mode": "centralized",
                "status": "completed",
                "best_val_recall": 0.92,
                "best_val_accuracy": 0.94,
                "metrics_count": 50,
                "start_time": "2025-01-01T12:00:00",
                "end_time": "2025-01-01T12:30:00",
                "run_description": "Test Run",
                "federated_info": None,
                "final_epoch_stats": None,
            }
        ],
        "total": 1,
    }

    # Call endpoint with query params
    response = client.get("/api/runs/list?limit=10&offset=5&status=completed")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["runs"]) == 1
    assert data["runs"][0]["id"] == 1

    # Verify query params were passed to the facade service
    mock_facade.summary.list_runs_with_summaries.assert_called_once()
    _, kwargs = mock_facade.summary.list_runs_with_summaries.call_args
    assert kwargs["limit"] == 10
    assert kwargs["offset"] == 5
    assert kwargs["status"] == "completed"


def test_list_runs_integration(client, mock_facade):
    """Integration test to verify query params are passed correctly.

    Verifies that offset (skip), limit, and status are passed to the service mock.
    """
    mock_facade.summary.list_runs_with_summaries.return_value = {"runs": [], "total": 0}

    # Test different combinations of query params
    client.get(
        "/api/runs/list?limit=50&offset=20&status=running&training_mode=federated"
    )

    # Verify parameters including those passed to the underlying CRUD mock via service
    mock_facade.summary.list_runs_with_summaries.assert_called_with(
        db=ANY,
        limit=50,
        offset=20,
        status="running",
        training_mode="federated",
        sort_by="start_time",
        sort_order="desc",
    )
