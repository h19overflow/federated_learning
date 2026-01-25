import pytest


@pytest.mark.unit
def test_get_run_detail_success(client, sample_run_data, mock_facade):
    """
    Positive test for run detail (metrics).
    GET /api/runs/{run_id}/metrics
    Assert 200 OK and correct ID.
    """
    # Configure mock to return valid data matching MetricsResponse schema
    mock_facade.metrics.get_run_metrics.return_value = {
        "experiment_id": f"run_{sample_run_data.id}",
        "status": "completed",
        "final_metrics": {"accuracy": 0.95, "loss": 0.1},
        "training_history": [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.1}],
        "total_epochs": 10,
        "metadata": {"mode": "centralized"},
    }

    response = client.get(f"/api/runs/{sample_run_data.id}/metrics")
    assert response.status_code == 200
    data = response.json()
    # MetricsResponse uses experiment_id as the field name
    assert data["experiment_id"] == f"run_{sample_run_data.id}"


@pytest.mark.unit
def test_get_run_detail_404(client, mock_facade):
    """
    Negative test for non-existent run.
    GET /api/runs/99999/metrics
    Assert 404 Not Found.
    """
    # Configure mock to raise ValueError (simulating not found)
    mock_facade.metrics.get_run_metrics.side_effect = ValueError("Run not found")

    response = client.get("/api/runs/99999/metrics")
    assert response.status_code == 404
