import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from federated_pneumonia_detection.src.api.main import app
from federated_pneumonia_detection.src.api.deps import (
    get_db,
    get_analytics,
    get_experiment_crud,
)
from federated_pneumonia_detection.src.control.analytics.facade import AnalyticsFacade
from federated_pneumonia_detection.src.boundary.CRUD.run import RunCRUD


@pytest.fixture
def mock_facade():
    """Mock AnalyticsFacade for testing."""
    mock = MagicMock(spec=AnalyticsFacade)
    mock.metrics = MagicMock()
    mock.summary = MagicMock()
    mock.backfill = MagicMock()
    mock.export = MagicMock()
    return mock


@pytest.fixture
def mock_run_crud():
    """Mock RunCRUD for testing."""
    return MagicMock(spec=RunCRUD)


@pytest.fixture
def client(mock_facade, mock_run_crud):
    """FastAPI TestClient with overridden dependencies."""

    def override_get_db():
        return MagicMock()  # Mock Session

    def override_get_analytics():
        return mock_facade

    def override_get_experiment_crud():
        return mock_run_crud

    # Override dependencies
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_analytics] = override_get_analytics
    app.dependency_overrides[get_experiment_crud] = override_get_experiment_crud

    with TestClient(app) as test_client:
        yield test_client

    # Clean up
    app.dependency_overrides.clear()
