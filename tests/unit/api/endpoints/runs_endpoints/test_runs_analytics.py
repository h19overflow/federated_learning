import pytest
from unittest.mock import MagicMock


@pytest.mark.unit
def test_get_analytics(api_client_with_db, sample_run_data):
    """
    Positive test for analytics summary.
    GET /api/runs/analytics/summary
    Assert 200 OK.
    """
    response = api_client_with_db.get("/api/runs/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_runs" in data
    assert "centralized" in data
    assert "federated" in data
    assert data["total_runs"] >= 1


def test_service_unavailable(api_client_with_db, mock_facade):
    """Test 503 error when analytics service is unavailable.

    Overrides get_analytics dependency to return None and asserts 503.
    """
    from federated_pneumonia_detection.src.api.deps import get_analytics
    from federated_pneumonia_detection.src.api.main import app

    # Force service to be None
    app.dependency_overrides[get_analytics] = lambda: None

    # Use existing analytics/summary endpoint which handles None facade
    response = api_client_with_db.get("/api/runs/analytics/summary")

    assert response.status_code == 503
    assert "Analytics service unavailable" in response.json()["detail"]

    # Clean up override for this specific test
    app.dependency_overrides.pop(get_analytics)
